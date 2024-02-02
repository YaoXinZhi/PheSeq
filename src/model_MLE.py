# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 07/09/2021 17:21
@Author: XINZHI YAO
"""
import math
import random

import numpy as np
import torch
import torch.nn as nn
from scipy.misc import derivative
from scipy.special import gamma


class evidence_encoder(torch.nn.Module):
    def __init__(self, _l_dim, _hidden_dim):
        super(evidence_encoder, self).__init__()

        self.hidden_nn = nn.Linear(_l_dim, _hidden_dim)

        self.beta_nn = nn.Linear(_hidden_dim, 2)

        self.relu = nn.ReLU()

    def forward(self, _lg):
        hidden_state = self.hidden_nn(_lg)
        # fixme: 9-7 sigmoid
        # _ag, _bg = self.beta_nn(hidden_state).split(1, dim=-1)
        # _ag = torch.sigmoid(_ag)
        # _bg = torch.sigmoid(_bg)

        _ag, _bg = self.relu(self.beta_nn(hidden_state)).split(1, dim=-1)

        _ag = _ag + 1e-4
        _bg = _bg + 1e-4

        return _ag, _bg

    def model_params(self):
        return [ *self.hidden_nn.parameters(), *self.beta_nn.parameters() ]


class GWAS_literature_MLE:

    def __init__(self, data_size: int, input_dim: int,
                 hidden_state_dim: int, time: int,
                 _phi_learning_rate=0.005,
                 threshold=0.05,
                 _gradient_accumulation=False):

        self.G = data_size
        self.threshold = threshold
        self.train_time = time
        self.phi_learning_rate = _phi_learning_rate
        self.gradient_accumulation = _gradient_accumulation

        self.alpha_G = [ 1 ] * self.G

        self.model = evidence_encoder(input_dim, hidden_state_dim)

        self.alpha_trained = False
        self.phi_trained = False

    def fake_data_generator(self, pos_mean=10, pos_var=1,
                            neg_mean=-10, neg_var=1,
                            pos_uni_lower=0, pos_uni_upper=0.05,
                            neg_uni_lower=0.1, neg_uni_upper=1):

        pos_pg_u = torch.distributions.Uniform(pos_uni_lower, pos_uni_upper)
        neg_pg_u = torch.distributions.Uniform(neg_uni_lower, neg_uni_upper)

        fake_L = [ ]
        for g in range(int(self.G / 2)):
            # positive data
            fake_L.append((np.random.normal(pos_mean, pos_var, (1, l_dim)), pos_pg_u.sample()))
            # negative data
            fake_L.append((np.random.normal(neg_mean, neg_var, (1, l_dim)), neg_pg_u.sample()))

        random.shuffle(fake_L)

        return fake_L

    def MLE_for_alpha(self, L_G: list):

        for g in range(self.G):
            fg, p_g = L_G[ g ]

            alpha_g_best = - (1 / math.log(p_g))

            self.alpha_G[g] = alpha_g_best

        self.alpha_trained = True

        print('Get best alpha_G though MLE.')

    @staticmethod
    def gamma_func(x):
        return gamma(x)

    def MLE_for_phi(self, L_G: list):

        if not self.alpha_trained:
            print('Alpha is not learned, please call model.MLE_for_alpha() first')

        print('Start training for phi update.')
        for t in range(self.train_time):
            # init gradient accumulation and zero gradient
            model_parameters = self.model.parameters()
            grad_acc = [ ]
            for para in model_parameters:
                grad_acc.append(torch.zeros_like(para))
            # G
            for g in range(self.G):

                lg, pg_true = L_G[g]
                lg = torch.from_numpy(lg).float()

                ag, bg = self.model(lg)

                ag_detach = ag.detach()
                bg_detach = bg.detach()

                f_g = torch.distributions.Beta(ag_detach, bg_detach).sample()

                # gradient for phi
                ag_gamma = self.gamma_func(ag_detach)
                bg_gamma = self.gamma_func(bg_detach)

                print(f'a: {ag_detach.item():.4f}, b:{bg_detach.item():.4f}')

                ag_bg_gamma = self.gamma_func(ag_detach + bg_detach)

                ag_gamma_der = derivative(self.gamma_func, ag_detach, dx=1e-6)
                bg_gamma_der = derivative(self.gamma_func, bg_detach, dx=1e-6)
                ag_bg_gamma_der = (derivative(self.gamma_func, ag_detach + bg_detach, dx=1e-6) + 1)

                grad_ag = math.log(f_g) \
                          + (ag_gamma * bg_gamma / ag_bg_gamma) \
                          * ((ag_bg_gamma_der * ag_gamma * bg_gamma) - (ag_bg_gamma * bg_gamma * ag_gamma_der)) \
                          /(math.pow(ag_gamma, 2) * math.pow(bg_gamma, 2))

                grad_bg = math.log(f_g) \
                          + (ag_gamma * bg_gamma / ag_bg_gamma) \
                          * ((ag_bg_gamma_der * ag_gamma * bg_gamma) - (ag_bg_gamma * ag_gamma * bg_gamma_der)) \
                          /(math.pow(ag_gamma, 2) * math.pow(bg_gamma, 2))


                grad_ag_phi = torch.autograd.grad(outputs=ag, inputs=model_parameters, grad_outputs=torch.ones_like(ag),
                                                  retain_graph=True)
                grad_bg_phi = torch.autograd.grad(outputs=bg, inputs=model_parameters, grad_outputs=torch.ones_like(bg),
                                                  allow_unused=True)

                # fixme: update each g
                if not self.gradient_accumulation:
                    for idx, (para, ag_para_grad, bg_para_grad) in enumerate(zip(model_parameters, grad_ag_phi, grad_bg_phi)):
                        grad = grad_ag * ag_para_grad + grad_bg * bg_para_grad
                        new_para = para + self.phi_learning_rate * grad
                        para.copy_(new_para)
                else:
                    # gradient accumulation
                    for idx, (ag_para_grad, bg_para_grad) in enumerate(zip(grad_ag_phi, grad_bg_phi)):
                        if grad_acc[idx].shape != ag_para_grad.shape \
                                or grad_acc[ idx ].shape != bg_para_grad.shape \
                                or ag_para_grad[ idx ].shape != bg_para_grad.shape:
                            raise ValueError(f'para.shape: {grad_acc[ idx ].shape}, grad.shape: {para_grad.shape}')
                        grad_acc[idx] += grad_ag * ag_para_grad + grad_bg * bg_para_grad

            # phi update
            if self.gradient_accumulation:
                with torch.no_grad():
                    for para, para_grad in zip(model_parameters, grad_acc):
                        if para.shape != para_grad.shape:
                            raise ValueError(f'para.shape: {para.shape}, grad.shape: {para_grad.shape}')
                        new_para = para + self.phi_learning_rate * para_grad / len(L_G)
                        para.copy_(new_para)

        self.phi_trained = True

        print('Get best phi though MLE.')

    def Evaluate(self, L_G: list, eval_count=None):

        if not self.alpha_trained or not self.phi_trained:
            raise ValueError('The model has not been trained.')

        if len(L_G) != self.G:
            raise ValueError(f'len(L_G) != len(self.G).')

        if eval_count is None:
            eval_count = len(L_G)

        _pos_pred = []
        _neg_pred = []

        pos_count = 0
        neg_count = 0
        with torch.no_grad():
            self.model.eval()
            for g in range(eval_count):

                lg, pg_true = L_G[ g ]
                alpha_g = self.alpha_G[ g ]

                lg = torch.from_numpy(lg).float()
                ag, bg = self.model(lg)

                # print(ag, bg)
                ag_detach = ag.detach().item()
                bg_detach = bg.detach().item()
                # print(ag_detach)
                # print(bg_detach)
                #
                # input()
                f_g = torch.distributions.Beta(ag_detach, bg_detach).sample()

                t_g = torch.distributions.Bernoulli(f_g).sample()
                if t_g == 1:
                    # beta(alpha, 1)
                    pg_pred = torch.distributions.Beta(alpha_g, 1).sample()
                else:
                    # U(0, 1)
                    pg_pred = torch.distributions.Uniform(0, 1).sample()

                if pg_pred <= self.threshold:
                    _pos_pred.append({
                        'pg_true': pg_true,
                        'ag': ag_detach,
                        'bg': bg_detach,
                        'fg': f_g,
                        'tg': t_g,
                        'alpha_g': alpha_g,
                        'p_pred': pg_pred,
                    })
                    pos_count += 1
                else:
                    _neg_pred.append({
                        'pg_true': pg_true,
                        'ag': ag_detach,
                        'bg': bg_detach,
                        'fg': f_g,
                        'tg': t_g,
                        'alpha_g': alpha_g,
                        'p_pred': pg_pred,
                    })
                    neg_count += 1

                print(f'pg_true: {pg_true:.4f}, '
                      f'{"无关" if pg_true > self.threshold else "有关"}')

                print(f'a_g: {ag_detach:.4f}, '
                      f'b_g: {bg_detach:.4f}, \n'
                      f'f_g: {f_g.item():.4f}, '
                      f'T_g: {t_g.item()}, '
                      f'alpha_g: {alpha_g:.4f}, '
                      f'p_g_pred: {pg_pred.item():.4f}, '
                      f'{"预测无关" if pg_pred.item() > self.threshold else "预测有关"}')

                print()
        print(f'Pos_pred: {pos_count}/{self.G}, '
              f'Neg_pred: {neg_count}/{self.G}')

        _pos_pred = sorted(_pos_pred, key=lambda x: x['p_pred'])
        _neg_pred = sorted(_neg_pred, key=lambda x: x['p_pred'])
        return _pos_pred, _neg_pred


if __name__ == '__main__':
    G = 20
    l_dim = 20
    hidden_dim = 50
    phi_learning_rate = 3e-5
    train_time = 200
    gradient_accumulation = False

    gwas_threshold = 0.05

    # model = GWAS_literature_MLE(G, gwas_threshold)
    model = GWAS_literature_MLE(G, l_dim, hidden_dim, train_time,
                                phi_learning_rate, gwas_threshold,
                                gradient_accumulation)

    fake_data = model.fake_data_generator()

    model.MLE_for_alpha(fake_data)


    model.MLE_for_phi(fake_data)

    pos_pred, neg_pred = model.Evaluate(fake_data)
