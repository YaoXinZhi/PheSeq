# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 01/09/2022 9:36
@Author: XINZHI YAO
"""

import os
import logging
import argparse

import math
import random
import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

import torch
import torch.nn as nn
import torch.nn.functional as F

import sympy
from sympy import *

from sklearn.metrics import mean_absolute_error, mean_squared_error


class VAE_encoder(nn.Module):
    def __init__(self, embedding_size: int, hidden_dim=200, z_dim=20):
        super(VAE_encoder, self).__init__()

        self.encoder_linear = nn.Linear(embedding_size, hidden_dim)

        self.hidden_mu_linear = nn.Linear(hidden_dim, z_dim)
        self.hidden_log_var_linear = nn.Linear(hidden_dim, z_dim)

        self.z_hidden_linear = nn.Linear(z_dim, hidden_dim)
        self.decoder_linear = nn.Linear(hidden_dim, embedding_size)

    def encoder(self, L_g):
        hidden = F.relu(self.encoder_linear(L_g))
        return self.hidden_mu_linear(hidden), self.hidden_log_var_linear(hidden)

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decoder(self, z):
        hidden = F.relu(self.z_hidden_linear(z))
        return F.sigmoid(self.decoder_linear(hidden))

    def forward(self, L_g):
        mu, log_var = self.encoder(L_g)
        z = self.reparameterize(mu, log_var)
        L_g_reconstructed = self.decoder(z)
        return L_g_reconstructed, mu, log_var, z

    def model_params(self):
        return [ *self.encoder.parameters(),
                 *self.hidden_mu_linear.parameters(),
                 *self.hidden_log_var_linear.parameters(),
                 *self.z_hidden_linear.parameters(),
                 *self.decoder.parameters(),
                 ]

    def generation_params(self):
        return [ *self.z_hidden_linear.parameters(),
                 *self.decoder.parameters()]

    def inference_params(self):
        return [*self.encoder.parameters(),
                *self.hidden_mu_linear.parameters()]

    def mu_params(self):
        return [*self.hidden_mu_linear.parameters()]

    def log_var_params(self):
        return [*self.hidden_log_var_linear.parameters()]

class evidence_encoder(nn.Module):

    def __init__(self, embedding_dim:int, hidden_dim:int,
                 z_dim:int,
                 multi_hidden: bool):
        super(evidence_encoder, self).__init__()

        self.VAE_encoder = VAE_encoder(embedding_dim, hidden_dim, z_dim)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.multi_hidden = multi_hidden
        if multi_hidden:
            print('Multi Hidden Layer.')

        self.hidden_nn = nn.Linear(z_dim, hidden_dim)

        self.multi_hidden = nn.Sequential(
            nn.Linear(z_dim, 512),
            # nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            # nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.Linear(128, z_dim)
        )

        self.beta_nn = nn.Linear(_hidden_dim, 3)

        self.relu = nn.ReLU()

    def forward(self, L_g):

        lg_reconstructed, mu, log_var, z = self.VAE_encoder(L_g)


        if self.multi_hidden:
            hidden_state = self.multi_hidden(z)
        else:
            hidden_state = self.hidden_nn(z)

        a_g, b_g, alpha_g = self.relu(self.beta_nn(hidden_state)).split(1, dim=-1)

        # avoid zero for ag/bg/alpha_g
        a_g = a_g + 1e-10
        b_g = b_g + 1e-10
        alpha_g = alpha_g + 1e-10

        return a_g,b_g, alpha_g, lg_reconstructed, mu, log_var, z

    def model_params(self):
        return [*self.hidden_nn.parameters(), *self.beta_nn.parameters()]


class Dynamic_EMFAS:
    def __init__(self, data_size: int, embedding_size: int,
                 hidden_state_dim: int, z_dim: int,
                 train_time: int, phi_learning_rate: float,
                 gradient_accumulation: bool, save_log: bool,
                 log_save_path: str, log_save_prefix: str,
                 batch_size:int, random_seed: int,
                 multi_hidden: bool, mu_learning_rate: float,
                 sigma_learning_rate: float):

        self.G = data_size
        self.embedding_dim = embedding_size
        self.hidden_state_dim = hidden_state_dim
        self.z_dim = z_dim

        self.multi_hidden = multi_hidden

        self.train_time = train_time
        self.phi_learning_rate = phi_learning_rate
        self.mu_learning_rate = mu_learning_rate
        self.sigma_learning_rate = sigma_learning_rate
        self.gradient_accumulation = gradient_accumulation

        self.save_log = save_log
        self.log_save_path = log_save_path
        self.log_save_prefix = log_save_prefix

        self.batch_size = batch_size
        self.random_seed = random_seed

        self.T, self.a, self.b, self.alpha, self.P, self.F, self.C = None, None, None, None, None, None, None
        self.objective_function = None

        # saved for MAP algorithm
        self.ag_list = []
        self.bg_list = []
        self.Tg_list = []

        self.logger = None

        self.logger_init()

        self.seed_init()

        self.model = evidence_encoder(self.embedding_dim, self.hidden_state_dim,
                                      self.z_dim, self.multi_hidden)

        self.MLE_objective_function_init()


    def seed_init(self):
        self.logger.info(f'Random seed: {self.random_seed}')
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)


    def logger_init(self):
        if not os.path.exists(self.log_save_path):
            os.mkdir(self.log_save_path)

        self.logger = logging.getLogger(__name__)
        if self.save_log:
            print('Save log file.')
            logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                datefmt='%m/%d/%Y %H:%M:%S',
                                level=logging.INFO,
                                filename=f'{self.log_save_path}/{self.log_prefix}.log',
                                filemode='w')
        else:
            print('Do not save log file.')
            logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                datefmt='%m/%d/%Y %H:%M:%S',
                                level=logging.INFO, )
    def MLE_objective_function_init(self):
        self.T, self.a, self.b, self.alpha, self.P, self.F, self.C = sympy.symbols('T a b alpha P F C')

        self.objective_function = self.T * sympy.log(self.alpha) \
                                  + self.T * (self.alpha - 1) * sympy.log(self.P) \
                                  + (self.T + self.a - 1) * sympy.log(self.F) \
                                  + (self.b - self.T) * sympy.log(1 - self.F) \
                                  + sympy.log(sympy.gamma(self.a + self.b) / sympy.gamma(self.a) / sympy.gamma(self.b)) + self.C

    def read_entrez_p(self, entrez_p_file: str):
        entrez_to_p = {}
        entrez_to_symbol = {}
        entrez_bag = defaultdict(list)
        entrez_tag = defaultdict(list)
        with open(entrez_p_file) as f:
            for line in f:
                l = line.strip().split('\t')
                if line.startswith('GENE_LINE'):
                    symbol, entrez, p = l[ 1 ], l[ 2 ], float(l[ -1 ])
                    entrez_to_p[ entrez ] = p
                    entrez_to_symbol[ entrez ] = symbol
                else:
                    pmid = l[ 0 ]
                    sentence = l[ 1 ]
                    tags = eval(l[ 2 ])

                    entrez_bag[ entrez ].append((pmid, sentence))
                    entrez_tag[ entrez ].append(tags)
        self.logger.info(f'data size: {len(entrez_to_p)}, \
                min_p: {min(entrez_to_p.values())}, \
                max_p: {max(entrez_to_p.values())}.')
        return entrez_to_p

    def load_summary_data(self, p_value_file: str, embedding_file: str,
                          only_p_list=False):

        self.logger.info('load entrez_p_file.')
        entrez_to_p = self.read_entrez_p(p_value_file)
        entrez_to_data = {}

        if only_p_list:
            p_list = sorted(list(entrez_to_p.values()))
            return p_list

        self.logger.info('loading entrez_embedding_file.')
        entrez_list, embedding_list = self.read_embedding(embedding_file)

        entrez_data = []
        for idx, entrez in enumerate(entrez_list):
            entrez_data.append((entrez, embedding_list[idx], torch.tensor(entrez_to_p[entrez])))
            entrez_to_data[entrez] = (entrez, embedding_list[idx], torch.tensor(entrez_to_p[entrez]))
        random.shuffle(entrez_data)
        self.logger.info(f'data size: {len(entrez_data)}')

        return entrez_data, entrez_to_p, entrez_to_data

    def fake_data_generator(self, pos_mean=10, pos_var=0.5,
                            neg_mean=0, neg_var=0.5,
                            pos_uni_lower=0, pos_uni_upper=0.05,
                            neg_uni_lower=0.1, neg_uni_upper=1):

        self.logger.info(f'Loading fake data.')
        pos_pg_u = torch.distributions.Uniform(pos_uni_lower, pos_uni_upper)
        neg_pg_u = torch.distributions.Uniform(neg_uni_lower, neg_uni_upper)

        fake_L = [ ]
        for g in range(int(self.G / 2)):
            # positive data
            fake_L.append((f'{g}-pos', np.random.normal(pos_mean, pos_var, (1, self.embedding_dim)), pos_pg_u.sample()))
            # negative data
            fake_L.append((f'{g}-neg', np.random.normal(neg_mean, neg_var, (1, self.embedding_dim)), neg_pg_u.sample()))

        random.shuffle(fake_L)
        self.logger.info(f'Fake data size: {len(fake_L)}')

        return fake_L

    def MAP_for_DL(self, pg_lg: list, return_dir: dir, time: int):

        pg_list, lg_list = [], []
        for pg, lg in pg_lg:
            pg_list.append(pg)
            lg_list.append(lg)

        # fg_list = return_dir['fg_list']
        lg_recon_list = return_dir['lg_recon_list']
        mu_list = return_dir['mu_list']
        log_var_list = return_dir['log_var_list']

        mu_paras = self.model.VAE_encoder.mu_params()
        sigma_paras = self.model.VAE_encoder.log_var_params()

        mu_grad_acc = []
        for para in self.model.VAE_encoder.mu_params():
            mu_grad_add.append(torch.zeros_like(para))

        sigma_grad_acc = []
        for para in self.model.VAE_encoder.log_var_params():
            sigma_grad_acc.append(para)

        _N = len(pg_list)
        for idx, pg in enumerate(pg_list):
            # lg = lg_list[idx]

            # block coordinate ascent
            Tg = self.Tg_list[idx]
            ag = self.ag_list[idx]
            bg = self.bg_list[idx]
            Fg = (Tg+ag-1)/(ag+bg-1)

            # back propagation
            lg_recon = lg_recon_list[idx]

            mu_g = mu_list[idx]
            log_var = log_var_list[idx]
            # ∇mu_g(log p(Lg_reconstructed|Zg))
            grad_lg_mu = torch.autograd.grad(outputs=math.log(lg_recon), inputs=mu_g,
                                             grad_outputs=torch.ones_like(lg_recon),
                                             allow_unused=True, retain_graph=True)

            # ∇mu_g(log p(Fg|Zg))
            grad_fg_mu = torch.autograd.grad(outputs=math.log(Fg), inputs=mu_g,
                                             grad_outputs=torch.ones_like(Fg),
                                             allow_unused=True, retain_graph=True)

            # ∇sigma_g(log p(Lg_reconstructed|Zg))
            grad_lg_sigma = torch.autograd.grad(outputs=math.log(lg_recon), inputs=log_var,
                                                grad_outputs=torch.ones_like(lg_recon),
                                                allow_unused=True, retain_graph=True)
            # ∇sigma_g(log p(Fg|Zg))
            grad_fg_sigma = torch.autograd.grad(outputs=math.log(Fg), inputs=log_var,
                                                grad_outputs=torch.ones_like(Fg),
                                                allow_unused=True, retain_graph=False)



            # gradient accumulation
            for dim, (lg_mu_grad, fg_mu_grad, lg_sigma_grad, fg_sigma_grad) in enumerate(
                zip(grad_lg_mu, grad_fg_mu, grad_lg_sigma, grad_fg_sigma)):
                if lg_mu_grad.shape != mu_grad_acc[dim] \
                    or lg_sigma_grad.shape != sigma_grad_acc[dim] \
                    or fg_mu_grad.shape != mu_grad_acc[dim] \
                    or fg_sigma_grad.shape != sigma_grad_acc[dim]:
                    raise ValueError(f'para.shape: {sigma_grad_acc[dim].shape}, grad.shape: {lg_sigma_grad.shape}')
                mu_grad_acc[dim] += float(grad_lg_mu) + float(grad_fg_mu)
                sigma_grad_acc[dim] += float(grad_lg_sigma) + float(grad_fg_sigma)

        # parameters iteration
        with torch.no_grad():
            # parameters update for mu
            for para, para_grad in zip(mu_paras, mu_grad_acc):
                if para.shape != para_grad.shape:
                    raise ValueError(f'para.shape: {para.shape}, grad.shape: {para_grad.shape}')
                new_mu_para = para + self.mu_learning_rate * para_grad/ _N
                para.copy_(new_mu_para)

            # parameters update for sigma
            for para, para_grad in zip(sigma_paras, sigma_grad_acc):
                if para.shape != para_grad.shape:
                    raise ValueError(f'para.shape: {para.shape}, grad.shape: {para_grad.shape}')
                new_sigma_para = para + self.sigma_learning_rate * para_grad / _N -0.5*(-para_grad.shape[1]/_N + para_grad.shape[1])
                para.copy_(new_sigma_para)
        print(f'time: {time}, deep learning parameter is updated.')


    def evaluate(self, eval_data_set: list, time: int, final_eval: bool,
                 entrez_to_alpha=None, use_best_alpha=False, smooth_mode=False):
        # reconstruction error
        self.logger.info('start evaluate.')
        pred_p_list = []
        ture_p_list = []
        t_list = []
        f_list = []

        entrez_list = []

        fg_list = []
        pg_list = []
        with torch.no_grad():
            for g in range(len(eval_data_set)):
                entrez_g, lg, pg_ture = eval_data_set[ g ]

                lg = torch.from_numpy(lg).float()

                ag, bg, alpha_g = self.model(lg)

                ag_detach = ag.detach()
                bg_detach = bg.detach()
                alpha_g_detach = alpha_g.detach()

                if use_best_alpha:
                    alpha_g_detach = torch.tensor(float(entrez_to_alpha[entrez_g]))

                fg = torch.distributions.Beta(ag_detach, bg_detach).sample()

                Tg = torch.distributions.Bernoulli(fg).sample()
                if Tg == 1:
                    # beta(alpha, 1)
                    pg_pred = torch.distributions.Beta(alpha_g_detach, 1).sample()
                else:
                    pg_pred = torch.distributions.Uniform(0, 1).sample()

                fg_list.append(fg.detach().item())
                pg_list.append(pg_ture.item())

                self.logger.info(f'Evaluation\tTime: {time}\tDataIdx:{g}\t'
                            f'ag-{ag_detach.item()}\tbg-{bg_detach.item()}\t'
                            f'alpha_g-{alpha_g_detach.item()}\t'
                            f'fg-{fg.item()}\ttg-{Tg.item()}\t'
                            f'p_pred-{pg_pred.item()}\tp_ture-{pg_ture}')

                entrez_list.append(entrez_g)
                pred_p_list.append(pg_pred.item())
                ture_p_list.append(pg_ture.item())
                t_list.append(Tg.item())
                f_list.append(fg.item())

        MAE = mean_absolute_error(ture_p_list, pred_p_list)
        MSE = mean_squared_error(ture_p_list, pred_p_list)

        self.logger.info(f'EvaluationResult\tTime-{time}\tMAE-{MAE}\tMSE-{MSE}')
        self.logger.info('')

        if final_eval:
            if smooth_mode:
                pred_save_file = f'{self.log_save_path}/{self.log_prefix}.smooth.p-pred.tsv'
            else:
                pred_save_file = f'{self.log_save_path}/{self.log_prefix}.{time}.p-pred.tsv'

            with open(pred_save_file, 'w') as wf:
                wf.write(f'Entrez\tF_g\tT_g\tTure P-value\tPred P-value\n')
                for g, entrez in enumerate(entrez_list):
                    wf.write(f'{entrez}\t{f_list[g]}\t{t_list[g]}\t{ture_p_list[g]}\t{pred_p_list[g]}\n')

            fg_pg_save_file = f'{self.log_save_path}/fg_pg.tsv'
            with open(fg_pg_save_file, 'w') as wf:
                for fg, pg in zip(fg_list, pg_list):
                    wf.write(f'{fg}\t{pg}\n')

            self.logger.info(f'{pred_save_file} save done.')
            return fg_list, pg_list

        # other GWAS recall ... (trick)

    def MLE_for_PGM(self, pg_lg: list, time: int):


        pg_list, lg_list = [], []
        for pg, lg in pg_lg:
            pg_list.append(pg)
            lg_list.append(lg)

        eval_data_list = pg_lg.copy()
        training_data_list = pg_lg.copy()

        ag_t_list = []
        bg_t_list = []
        alpha_g_t_list = []

        ag_quo_list = []
        bg_quo_list = []
        alpha_quo_list = []

        self.ag_list = []
        self.bg_list = []
        self.Tg_list = []

        model_parameters = self.model.model_params()

        early_stop = False
        grad_acc = []
        for para in model_parameters:
            grad_acc.append(torch.zeros_like(para))

        # G iteration
        fg_list = []
        pg_list = []

        lg_recon_list = []
        mu_list = []
        log_var_list = []
        z_list = []

        wf_ag_quo = open(f'{self.log_save_path}/ag_quo.txt', 'w')
        wf_bg_quo = open(f'{self.log_save_path}/bg_quo.txt', 'w')
        wf_alpha_quo = open(f'{self.log_save_path}/alpha_g_quo.txt', 'w')

        for g in range(len(training_data_list)):
            if g % int(len(training_data_list) / 10) == 0:
                logging.info(f'Training:\ttime: {time}, g: {g}/{len(training_data_list)} training done.')

            entrez_g, lg, pg_ture = training_data_list[ g ]
            if pg_ture == 0:
                pg_ture = torch.tensor(1e-30)

            lg = torch.from_numpy(lg).float()

            # single number now
            a_g,b_g, alpha_g, lg_reconstructed, mu, log_var, z = self.model(lg)

            lg_recon_list.append(lg_reconstructed)
            mu_list.append(mu)
            log_var_list.append(log_var_list)
            z_list.append(z)

            ag_detach = ag.detach()
            bg_detach = bg.detach()
            alpha_g_detach = alpha_g.detach()

            fg = torch.distributions.Beta(ag_detach, bg_detach).sample()

            Tg = torch.distributions.Bernoulli(fg).sample()

            # fg_list.append(fg.detach().item())
            # pg_list.append(pg_ture.item())
            fg_list.append(fg.item())
            pg_list.append(pg_ture.item())

            self.ag_list.append(ag_detach)
            self.bg_list.append(bg_detach)
            self.Tg_list.append(Tg)

            # ∇αg
            grad_alpha_g = diff(self.objective_function, self.alpha).evalf(
                subs={'T': Tg, 'alpha': alpha_g_detach, 'P': pg_ture})

            # ∇ag
            grad_ag = diff(self.objective_function, self.a).evalf(subs={'a': ag_detach, 'b': bg_detach, 'F': fg})

            # ∇bg
            grad_bg = diff(self.objective_function, self.b).evalf(subs={'a': ag_detach, 'b': bg_detach, 'F': fg})

            # ∂ag/∂φ
            grad_ag_phi = torch.autograd.grad(outputs=ag, inputs=model_parameters,
                                              grad_outputs=torch.ones_like(ag),
                                              retain_graph=True)

            # ∂bg/∂φ
            grad_bg_phi = torch.autograd.grad(outputs=bg, inputs=model_parameters,
                                              grad_outputs=torch.ones_like(bg),
                                              allow_unused=True, retain_graph=True)

            # ∂αg/∂φ
            grad_alpha_g_phi = torch.autograd.grad(outputs=alpha_g, inputs=model_parameters,
                                                   grad_outputs=torch.ones_like(alpha_g),
                                                   allow_unused=True, )
            if save_each_step:
                plot_save_file = f'{self.log_save_path}/{self.log_prefix}.Time-{time}.Step-{g}.pg-fg.png'
                self.save_fg_pg_plot(pg_list, fg_list, plot_save_file)

            # save ag/bg/alpha_g for quotient
            if time == 0:
                ag_t_list.append(ag_detach.item())
                bg_t_list.append(bg_detach.item())
                alpha_g_t_list.append(alpha_g_detach.item())

                ag_quo_list.append(0)
                bg_quo_list.append(0)
                alpha_quo_list.append(0)
            else:
                ag_quo = ag_detach.item() / ag_t_list[ g ]
                bg_quo = bg_detach.item() / bg_t_list[ g ]
                alpha_g_quo = alpha_g_detach.item() / alpha_g_t_list[ g ]

                ag_t_list[ g ] = ag_detach.item()
                bg_t_list[ g ] = bg_detach.item()
                alpha_g_t_list[ g ] = alpha_g_detach.item()

                ag_quo_list[ g ] = ag_quo
                bg_quo_list[ g ] = bg_quo
                alpha_quo_list[ g ] = alpha_g_quo

            # gradient accumulation
            if self.gradient_accumulation:
                for idx, (ag_para_grad, bg_para_grad, alpha_g_para_grad) in enumerate(
                        zip(grad_ag_phi, grad_bg_phi, grad_alpha_g_phi)):
                    if grad_acc[ idx ].shape != ag_para_grad.shape \
                            or grad_acc[ idx ].shape != bg_para_grad.shape \
                            or ag_para_grad.shape != bg_para_grad.shape \
                            or ag_para_grad.shape != alpha_g_para_grad.shape:
                        raise ValueError(f'para.shape: {grad_acc[ idx ].shape}, grad.shape: {ag_para_grad.shape}')
                    grad_acc[ idx ] += float(grad_ag) * ag_para_grad \
                                       + float(grad_bg) * bg_para_grad \
                                       + float(grad_alpha_g) * alpha_g_para_grad
            else:
                with torch.no_grad():
                    for idx, (para, ag_para_grad, bg_para_grad, alpha_g_para_grad) \
                            in enumerate(zip(model_parameters, grad_ag_phi, grad_bg_phi, grad_alpha_g_phi)):
                        if ag_para_grad.shape != bg_para_grad.shape \
                                or ag_para_grad.shape != alpha_g_para_grad.shape:
                            raise ValueError(
                                f'ag_para_grad.shape: {ag_para_grad.shape}, grad.shape: {ag_para_grad.shape}')

                        grad = float(grad_ag) * ag_para_grad \
                               + float(grad_bg) * bg_para_grad \
                               + float(grad_alpha_g) * alpha_g_para_grad

                        new_para = para + self.phi_learning_rate * grad
                        para.copy_(new_para)

        # save ag_quo/bg_quo/alpha_g_quo for observe convergence.
        ag_quo_wf = '\t'.join(map(str, ag_quo_list))
        bg_quo_wf = '\t'.join(map(str, bg_quo_list))
        alpha_quo_wf = '\t'.join(map(str, alpha_quo_list))

        wf_ag_quo.write(f'time: {time}, ag_quo: {ag_quo_wf}\n')
        wf_bg_quo.write(f'time: {time}, bg_quo: {bg_quo_wf}\n')
        wf_alpha_quo.write(f'time: {time}, alpha_quo: {alpha_quo_wf}\n')

        # early_stopping determine
        early_stop_bool, ag_quo_con_por, bg_quo_con_por, alpha_g_quo_con_por = self.early_stopping(ag_quo_list,
                                                                                                   bg_quo_list,
                                                                                                   alpha_quo_list)

        if early_stop_bool and use_early_stop:
            print(f'Early stopping, time: {time}.')
            self.logger.info(f'Early stopping, time: {time}.')
            self.logger.info(f'ag_convergence: {ag_quo_con_por:.2f}, '
                             f'bg_convergence: {bg_quo_con_por:.2f}, '
                             f'alpha_g_convergence: {alpha_g_quo_con_por:.2f}.')
            early_stop = True


        elif time > 10:
            print(f'Total training done, time: {time}.')
            self.logger.info(f'Total training done, time: {time}.')
            self.logger.info(f'ag_convergence: {ag_quo_con_por:.2f}, '
                             f'bg_convergence: {bg_quo_con_por:.2f}, '
                             f'alpha_g_convergence: {alpha_g_quo_con_por:.2f}.')

            early_stop = True
        else:
            self.logger.info(f'ag_convergence: {ag_quo_con_por:.2f}, '
                             f'bg_convergence: {bg_quo_con_por:.2f}, '
                             f'alpha_g_convergence: {alpha_g_quo_con_por:.2f}.')

        # gradient accumulation phi update
        if self.gradient_accumulation:
            with torch.no_grad():
                for para, para_grad in zip(model_parameters, grad_acc):
                    if para.shape != para_grad.shape:
                        raise ValueError(f'para.shape: {para.shape}, grad.shape: {para_grad.shape}')
                    new_para = para + self.phi_learning_rate * para_grad / len(training_data_list)
                    para.copy_(new_para)

            if save_each_epoch:
                self.evaluate(eval_data_list, time, True, entrez_to_alpha, use_best_alpha)
            else:
                self.evaluate(eval_data_list, time, False, None, False, True)

        self.logger.info('Final evaluation.')
        fg_list, pg_list = self.evaluate(eval_data_list, time, True,
                                         entrez_to_alpha, use_best_alpha, False)

        return_dir = {
            'fg_list': fg_list,
            'pg_list': pg_list,
            'lg_recon_list': lg_recon_list,
            'mu_list': mu_list,
            'log_var_list': log_var_list,
            'z_list': z_list,
            'easy_stop': early_stop,
        }


        return return_dir

    def model_train(self, pg_lg: list):
        """
        MAP for deep learning optimization and MLE for PGM optimization.
        1.  load the data. (per-loaded)
            for fixed disease d.
            read summary data for each gene g.
            pre-computed p-value for each gene g.
            pg_lg: {(entrez, p-value, [embedding]), ()...}
        2. model initialize. (per-loaded)
        3. MAP for deep learning.
        4. MLE for PGM.
        :return:
        """
        # based on Algorithm 2
        # for lg, pg in embedding_p_data:
        #     pass
        pg_list = []
        lg_list = []
        for pg, lg in pg_lg:
            pg_list.append(pg)
            lg_list.append(lg)

        for epoch in range(self.train_time):
            # Initialize the latent variable
            print(f'epoch: {epoch}')
            return_dir = self.MLE_for_PGM(pg_lg, epoch)
            self.MAP_for_DL(pg_lg, return_dir, epoch)

            early_stop = return_dir['early_stop']
            if early_stop:
                print(f'early stop: {epoch}')



def main():
    parser = argparse.ArgumentParser(description='Dynamic EMFAS model.')

    parser.add_argument('-mf', dest='summary_file', default='',
                        help='summary_file including p-value and association description for each gene.')

    parser.add_argument('-ef', dest='embedding_file', default='',
                        help='embedding representative data for each gene with fixed disease.')

    parser.add_argument('-ds', dest='data_size', type=int,
                        default=100,
                        help='default: 100')

    parser.add_argument('-ed', dest='embedding_size', type=int,
                        default='128',
                        help='default: 128')

    parser.add_argument('-mh', dest='multi_hidden', action='store_true',
                        default=False,
                        help='multi_hidden, default: False')
    parser.add_argument('-hs', dest='hidden_dim', type=int,
                        default=24)
    parser.add_argument('-zd', dest='z_dim', type=int,
                        default=12)
    parser.add_argument('-bs', dest='batch_size', type=int,
                        default=128)

    parser.add_argument('-tt', dest='train_time', type=int,
                        default=100)
    parser.add_argument('-lr', dest='learning_rate', type=float,
                        default=0.005)

    parser.add_argument('-mr', dest='mu_learning_rate', type=float,
                        default=0.005)
    parser.add_argument('-vr', dest='sigma_learning_rate', type=float,
                        default=0.005)

    parser.add_argument('-rs', dest='random_seed', type=int,
                        default=126,
                        help='default: 126')
    parser.add_argument('-ga', dest='gradient_accumulation', action='store_false',
                        default=True,
                        help='gradient_accumulation, default=True')

    parser.add_argument('-sl', dest='save_log', action='store_true',
                        default=False,
                        help='save_log, default: False')
    parser.add_argument('-lp', dest='log_save_path', default='../log',
                        help='default ../log')
    parser.add_argument('-lp', dest='log_save_prefix', default='dynamic_emfas')

    parser.add_argument('-uf', dest='use_fake_data', action='store_true',
                        default=False,
                        help='use_fake_data, default: False')

    args = parser.parse_args()

    model = Dynamic_EMFAS(args.data_size, args.embedding_size, args.hidden_dim, args.z_dim,
                          args.train_time, args.learning_rate, args.gradient_accumulation,
                          args.save_log, args.log_save_path, args.log_save_prefix,
                          args.batch_size, args.random_seed, args.multi_hidden, args.mu_learning_rate,
                          args.sigma_learning_rate)
    

    if args.use_fake_data:
        print('Fake data from nihility.')
        data = model.fake_data_generator()
    else:
        data = model.load_summary_data(args.summary_file, args.embedding_file)

    model.model_train(data)

if __name__ == '__main__':
    main()

