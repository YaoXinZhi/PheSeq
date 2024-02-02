# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 27/09/2021 22:31
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

import sympy
from sympy import *

from sklearn.metrics import mean_absolute_error, mean_squared_error


class evidence_encoder(torch.nn.Module):
    def __init__(self, _l_dim, _hidden_dim, multi_hidden):
        super(evidence_encoder, self).__init__()

        self.multi_hidden = multi_hidden
        if multi_hidden:
            print('Multi Hidden Layer.')

        self.hidden_nn = nn.Linear(_l_dim, _hidden_dim)

        self.multi_hidden = nn.Sequential(
            nn.Linear(_hidden_dim, 512),
            # nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            # nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.Linear(128, _hidden_dim)
        )

        self.beta_nn = nn.Linear(_hidden_dim, 3)

        # self.alpha_nn = nn.Linear(_hidden_dim, 1)

        self.relu = nn.ReLU()

    def forward(self, _lg):
        hidden_state = self.hidden_nn(_lg)

        if self.multi_hidden:
            hidden_state = self.multi_hidden(hidden_state)

        _ag, _bg, _alpha_g = self.relu(self.beta_nn(hidden_state)).split(1, dim=-1)
        # _alpha_g = self.relu(self.alpha_nn)

        # avid zero for ag/bg/alpha_g
        _ag = _ag + 1e-10
        _bg = _bg + 1e-10
        _alpha_g = _alpha_g + 1e-10

        return _ag, _bg, _alpha_g

    def model_params(self):
        return [ *self.hidden_nn.parameters(), *self.beta_nn.parameters() ]

class GWAS_literature_MLE:

    def __init__(self, data_size: int, input_dim: int,
                 hidden_state_dim: int, train_time: int,
                 phi_learning_rate=0.005,
                 significant_threshold=0.05,
                 gradient_accumulation=False,
                 save_log=False,
                 log_save_path='../log',
                 log_save_prefix='test',
                 batch_size=128,
                 random_seed=126,
                 multi_hidden=False,
                 ):

        self.G = data_size
        self.input_dim = input_dim
        self.significant_threshold = significant_threshold
        self.train_time = train_time
        self.phi_learning_rate = phi_learning_rate
        self.gradient_accumulation = gradient_accumulation
        self.batch_size = batch_size
        self.random_seed = random_seed


        self.save_log = save_log
        self.log_save_path = log_save_path
        self.log_prefix = log_save_prefix

        # Pythonic
        self.T, self.a, self.b, self.alpha, self.P, self.F = None, None, None, None, None, None
        self.objective_function = None
        self.logger = None

        self.alpha_trained = False
        self.phi_trained = False

        self.logger_init()

        self.seed_init()

        self.model = evidence_encoder(input_dim, hidden_state_dim, multi_hidden)

        self.objective_function_init()

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

    # @staticmethod
    def read_embedding(self, embedding_file: str):
        embedding_list = [ ]
        entrez_list = [ ]
        with open(embedding_file) as f:
            for line in f:
                l = line.strip().split('\t')
                if len(l) != 2:
                    print(l)
                    input()
                entrez_list.append(l[ 0 ])

                embedding = list(map(float, l[ 1 ].split()))
                embedding_list.append(embedding)
        self.logger.info(f'data size: {len(entrez_list):,}, feature size: {len(embedding_list[ 0 ])}')
        return entrez_list, np.array(embedding_list)

    def load_batch_data(self, data: list):
        self.logger.info('loading batch data.')
        batch_data_list = []

        start = 0
        while True:
            if start > len(batch_data_list):
                break
            batch_data = data[start: start+self.batch_size]
            batch_data_list.append(batch_data)
            start += self.batch_size
        self.logger.info(f'loading batch data: {len(batch_data_list)} batches.')
        return batch_data_list

    # @staticmethod
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
        # return entrez_to_p, entrez_bag, entrez_to_symbol, entrez_tag
        return entrez_to_p

    @staticmethod
    def read_entrez_to_best_alpha(entrez_to_alpha_file: str):

        entrez_to_alpha = {}
        with open(entrez_to_alpha_file) as f:
            f.readline()
            for line in f:
                entrez, alpha = line.strip().split('\t')
                entrez_to_alpha[entrez] = float(alpha)
        return entrez_to_alpha

    def load_data(self, p_value_file: str, embedding_file: str,
                  only_p_list=False):
        self.logger.info('loading entrez_p_file.')
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

    def fake_data_generate_from_real_data(self, p_list: list,
                                          mu_factor: float, sigma: float,
                                          simple_p:bool, uniform_p:bool,
                                          log_p:bool, distinctive_normal:bool,
                                          distinctive_threshold: float):

        self.logger.info(f'Loading fake data from real data.')
        self.logger.info(f'mu_factor: {mu_factor}, sigma: {sigma}')
        gap = int(len(p_list) / self.G)
        fake_p_list = [ p for idx, p in enumerate(p_list) if idx % gap == 0 ]

        if uniform_p:
            print('uniform p-value.')
            fake_p_list = np.linspace(1, 0, self.G, endpoint=False).tolist()

        if log_p:
            print('use log-p')

        if simple_p:
            print('simple p-value')
        else:
            print('normal p-value')
            if distinctive_normal:
                print('Distinctive Normal.')

        fake_L = [ ]
        for idx, p in enumerate(fake_p_list):

            if p == 0:
                p = 1e-300

            if log_p:
                embedding_p = math.log(p)
            else:
                embedding_p = p

            if simple_p:
                embedding = np.array([embedding_p]*self.input_dim)
            else:
                if distinctive_normal:
                    if embedding_p < distinctive_threshold:
                        mu = embedding_p * 200
                    else:
                        mu = embedding_p * -200
                else:
                    mu = embedding_p * mu_factor
                embedding = np.random.normal(mu, sigma, self.input_dim)

            fake_L.append((f'{idx}-idx', embedding, torch.tensor(p)))

        fake_L = fake_L[:self.G]

        random.shuffle(fake_L)
        self.logger.info(f'data size: {len(fake_p_list)}')
        return fake_L

    def fake_data_generator(self, pos_mean=10, pos_var=1,
                            neg_mean=-10, neg_var=1,
                            pos_uni_lower=0, pos_uni_upper=0.05,
                            neg_uni_lower=0.1, neg_uni_upper=1):

        self.logger.info(f'Loading fake data.')
        pos_pg_u = torch.distributions.Uniform(pos_uni_lower, pos_uni_upper)
        neg_pg_u = torch.distributions.Uniform(neg_uni_lower, neg_uni_upper)

        fake_L = [ ]
        for g in range(int(self.G / 2)):
            # positive data
            fake_L.append((f'{g}-pos', np.random.normal(pos_mean, pos_var, (1, self.input_dim)), pos_pg_u.sample()))
            # negative data
            fake_L.append((f'{g}-neg', np.random.normal(neg_mean, neg_var, (1, self.input_dim)), neg_pg_u.sample()))

        random.shuffle(fake_L)
        self.logger.info(f'Fake data size: {len(fake_L)}')

        return fake_L

    def objective_function_init(self):

        self.T, self.a, self.b, self.alpha, self.P, self.F = sympy.symbols('T a b alpha P F')

        self.objective_function = self.T*sympy.log(self.alpha) \
                 + self.T * (self.alpha - 1) * sympy.log(self.P) \
                 + (self.T+self.a-1)*sympy.log(self.F) \
                 + (self.b-self.T)*sympy.log(1-self.F) \
                 + sympy.log(sympy.gamma(self.a+self.b)/sympy.gamma(self.a)/sympy.gamma(self.b))


    @staticmethod
    def early_stopping(ag_quo_list: list, bg_quo_list: list, alpha_g_quo_list: list):

        ag_quo_con_por = len([ag for ag in ag_quo_list if 0.95 < ag < 1.2 ]) / len(ag_quo_list)
        bg_quo_con_por = len([bg for bg in bg_quo_list if 0.95 < bg < 1.2 ]) / len(bg_quo_list)
        alpha_g_quo_con_por = len([ alpha_g for alpha_g in alpha_g_quo_list if 0.95 < alpha_g < 1.2 ]) / len(alpha_g_quo_list)

        if ag_quo_con_por > 0.9 and bg_quo_con_por > 0.9 and alpha_g_quo_con_por > 0.9:
            return True, ag_quo_con_por, bg_quo_con_por, alpha_g_quo_con_por
        else:
            return False, ag_quo_con_por, bg_quo_con_por, alpha_g_quo_con_por

    @staticmethod
    def subsampling(data_list: str, sample_count: int):
        if len(data_list) < sample_count:
            raise ValueError(f'data size(len(data_list)) must larger than sample count.')

        return random.sample(data_list, sample_count)

    @staticmethod
    def upsampling(data_list: str, sample_count: int):
        if len(data_list) > sample_count:
            raise ValueError(f'data size(len(data_list)) must smaller than sample count.')

        return np.random.choice(data_list, sample_count)


    @staticmethod
    def scope_count(scope: int, data_list: list):
        scope = scope + 1

        scope_count_dict = {}

        scope_list = np.linspace(0, 1, scope)
        for idx in range(len(scope_list)):
            if idx == len(scope_list) - 1:
                break
            scope_count_dict[ (scope_list[ idx ], scope_list[ idx + 1 ]) ] = 0

        scope_to_entrez_list = defaultdict(list)
        for data in data_list:
            # print(data[0])
            entrez = data[0]
            p = data[2].item()
            for (start, end) in scope_count_dict.keys():
                if start <= p < end:
                    scope_count_dict[ (start, end) ] += 1
                    scope_to_entrez_list[(start, end)].append(entrez)

        return scope_count_dict, scope_to_entrez_list

    def sub_up_sampling(self, scope_count_dict, scope_to_entrez_list,
                        training_data_size, scope,
                        entrez_to_data):
        scope_data_size = int(training_data_size / scope)
        new_entrez_list = []
        for (start, end), entrez_list in scope_to_entrez_list.items():
            if scope_count_dict[ (start, end) ] == 0:
                continue
            elif scope_count_dict[ (start, end) ] > scope_data_size:
                sample_data_list = self.subsampling(entrez_list, scope_data_size)
            elif scope_count_dict[ (start, end) ] < scope_data_size:
                sample_data_list = self.upsampling(entrez_list, scope_data_size)
            else:
                sample_data_list = entrez_list.copy()

            new_entrez_list.extend(sample_data_list)

        random.shuffle(new_entrez_list)

        new_data_list = [entrez_to_data[entrez] for entrez in new_entrez_list]

        self.logger.info(f'training_data_size: {len(new_entrez_list)}')

        return new_data_list


    def MLE_for_update(self, L_G: list,
                       entrez_to_p=None,
                       scope=10,
                       training_data_size=3600,
                       use_fake_data=False,
                       entrez_to_data=None,
                       save_each_epoch=False,
                       use_early_stop=True,
                       save_each_step=False):

        print('Start training for phi and alpha update.')
        self.logger.info('Start training for phi and alpha update.')

        eval_data_list = L_G.copy()

        if save_each_epoch and save_each_step:
            save_each_epoch = False

        if not use_fake_data and data_smoothing:
            print('Use data smoothing')
            self.logger.info('Use data smoothing.')
            scope_count_dict, scope_to_entrez_list = self.scope_count(scope, L_G)
            self.logger.info(f'old scope_count_dict: {scope_count_dict}')

            training_data_list = self.sub_up_sampling(scope_count_dict, scope_to_entrez_list,
                                                      training_data_size, scope,
                                                      entrez_to_data)

            new_scope_count_dict, _ = self.scope_count(scope, training_data_list)
            self.logger.info(f'new scope_count_dict: {new_scope_count_dict}')
        else:
            print('Do not use data smoothing')

            training_data_list = L_G.copy()


        entrez_to_alpha = self.get_best_alpha(entrez_to_p)


        ag_t_list = []
        bg_t_list = []
        alpha_g_t_list = []

        ag_quo_list = []
        bg_quo_list = []
        alpha_quo_list = []

        wf_ag_quo = open(f'{self.log_save_path}/ag_quo.txt', 'w')
        wf_bg_quo = open(f'{self.log_save_path}/bg_quo.txt', 'w')
        wf_alpha_quo = open(f'{self.log_save_path}/alpha_g_quo.txt', 'w')

        time = 0
        for time in range(self.train_time):

            # init gradient accumulation and zero gradient
            model_parameters = self.model.model_params()

            grad_acc = []
            for para in model_parameters:
                grad_acc.append(torch.zeros_like(para))

            # G
            fg_list = []
            pg_list = []
            for g in range(len(training_data_list)):
                if g % int(len(training_data_list)/10) == 0:
                    logging.info(f'Training:\ttime: {time}, g: {g}/{len(training_data_list)} training done.')

                entrez_g, lg, pg_ture = training_data_list[g]
                if pg_ture == 0:
                    pg_ture = torch.tensor(1e-30)

                lg = torch.from_numpy(lg).float()

                # single number now
                ag, bg, alpha_g = self.model(lg)

                ag_detach = ag.detach()
                bg_detach = bg.detach()
                alpha_g_detach = alpha_g.detach()

                fg = torch.distributions.Beta(ag_detach, bg_detach).sample()

                Tg = torch.distributions.Bernoulli(fg).sample()

                fg_list.append(fg.detach().item())
                pg_list.append(pg_ture.item())

                # ∇αg
                grad_alpha_g = diff(self.objective_function, self.alpha).evalf(subs = {'T': Tg, 'alpha': alpha_g_detach, 'P': pg_ture})

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
                                                     allow_unused=True,)
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
                    for idx, (ag_para_grad, bg_para_grad, alpha_g_para_grad) in enumerate(zip(grad_ag_phi, grad_bg_phi, grad_alpha_g_phi)):
                        if grad_acc[idx].shape != ag_para_grad.shape \
                                or grad_acc[idx].shape != bg_para_grad.shape \
                                or ag_para_grad.shape != bg_para_grad.shape\
                                or ag_para_grad.shape != alpha_g_para_grad.shape:
                            raise ValueError(f'para.shape: {grad_acc[idx].shape}, grad.shape: {ag_para_grad.shape}')
                        grad_acc[idx] += float(grad_ag) * ag_para_grad \
                                           + float(grad_bg) * bg_para_grad \
                                           + float(grad_alpha_g) * alpha_g_para_grad
                else:
                    with torch.no_grad():
                        for idx, (para, ag_para_grad, bg_para_grad, alpha_g_para_grad) \
                                in enumerate(zip(model_parameters, grad_ag_phi, grad_bg_phi, grad_alpha_g_phi)):
                            if ag_para_grad.shape != bg_para_grad.shape \
                                or ag_para_grad.shape != alpha_g_para_grad.shape:
                                raise ValueError(f'ag_para_grad.shape: {ag_para_grad.shape}, grad.shape: {ag_para_grad.shape}')

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
            early_stop_bool, ag_quo_con_por, bg_quo_con_por, alpha_g_quo_con_por = self.early_stopping(ag_quo_list, bg_quo_list, alpha_quo_list)

            if early_stop_bool and use_early_stop:
                print(f'Early stopping, time: {time}.')
                self.logger.info(f'Early stopping, time: {time}.')
                self.logger.info(f'ag_convergence: {ag_quo_con_por:.2f}, '
                                 f'bg_convergence: {bg_quo_con_por:.2f}, '
                                 f'alpha_g_convergence: {alpha_g_quo_con_por:.2f}.')

                break
            elif time > 10:
                print(f'Total training done, time: {time}.')
                self.logger.info(f'Total training done, time: {time}.')
                self.logger.info(f'ag_convergence: {ag_quo_con_por:.2f}, '
                                 f'bg_convergence: {bg_quo_con_por:.2f}, '
                                 f'alpha_g_convergence: {alpha_g_quo_con_por:.2f}.')

                break
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
                fg_list, pg_list = self.evaluate(eval_data_list, time, entrez_to_alpha, True)
                plot_save_file = f'{self.log_save_path}/{self.log_prefix}.{time}.pg-fg.png'
                self.save_fg_pg_plot(pg_list, fg_list, plot_save_file)
            else:
                self.evaluate(eval_data_list, time, None, False, True)

        self.logger.info('Final evaluation.')
        fg_list, pg_list = self.evaluate(eval_data_list, time,
                                         entrez_to_alpha, True, False)
        return fg_list, pg_list

    @staticmethod
    def get_best_alpha(entrez_to_p: dict):
        entrez_to_alpha = {}
        for entrez, p in entrez_to_p.items():
            if p == 0:
                p = 1e-128

            alpha_g_best = - (1 / math.log(p))
            entrez_to_alpha[entrez] = alpha_g_best

        return entrez_to_alpha


    def evaluate(self, eval_data_set: list, time: int, final_eval: bool,
                 entrez_to_alpha=None, smooth_mode=False):
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

    def save_fg_pg_plot(self, pg_list: list, fg_list: list, plot_save_file: str):

        self.logger.info(f'Saving pg-fg scatter.')

        fig = plt.figure(figsize=(100, 2))

        x_major_locator = MultipleLocator(0.01)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)

        plt.scatter(pg_list, fg_list)

        fig.savefig(plot_save_file)
        self.logger.info(f'{plot_save_file} save done.')


def main():

    parser = argparse.ArgumentParser(description='Lit-GWAS Bayes model.')

    parser.add_argument('-ef', dest='embedding_file', type=str,
                        required=True)

    parser.add_argument('-sf', dest='summary_data_file', type=str,
                        required=True)

    parser.add_argument('-sl', dest='save_log', action='store_true',
                        default=False,
                        help='save_log, default: False')

    parser.add_argument('-lp', dest='log_save_path', default='../log',
                        help='default ../log')

    parser.add_argument('-lf', dest='log_prefix', default='predict',
                        help='predict')

    parser.add_argument('-ga', dest='gradient_accumulation', action='store_false',
                        default=True,
                        help='gradient_accumulation, default=True')

    parser.add_argument('-mh', dest='multi_hidden', action='store_true',
                        default=False,
                        help='multi_hidden, default: False')

    parser.add_argument('-rs', dest='random_seed', type=int,
                        default=126,
                        help='default: 126')

    parser.add_argument('-sm', dest='data_smoothing', action='store_true',
                        default=False,
                        help='data_smoothing, default: False')

    parser.add_argument('-st', dest='smoothing_training_data_size', type=int,
                        default=3600,
                        help='default: 4600')

    parser.add_argument('-ss', dest='smoothing_scope', type=int,
                        default=10,
                        help='default: 10')

    parser.add_argument('-uf', dest='use_fake_data', action='store_true',
                        default=False,
                        help='use_fake_data, default: False')
    
    parser.add_argument('-ds', dest='fake_data_size', type=int,
                        default=100,
                        help='default: 100')

    parser.add_argument('-ed', dest='embedding_size', type=int,
                        default='128',
                        help='default: 128')

    parser.add_argument('-lr', dest='learning_rate', type=float,
                        default=0.005)

    parser.add_argument('-hd', dest='hidden_dim', type=int,
                        default=50)

    parser.add_argument('-tt', dest='train_time', type=int,
                        default=100)

    parser.add_argument('-bs', dest='batch_size', type=int,
                        default=128)

    parser.add_argument('-pt', dest='p_value_threshold', type=float,
                        default=0.05)

    args = parser.parse_args()

    random_seed = args.random_seed

    hidden_dim = args.hidden_dim

    phi_learning_rate = args.learning_rate
    train_time = args.train_time

    batch_size = args.batch_size

    gwas_threshold = args.p_value_threshold

    save_log = args.save_log
    log_save_path = args.log_save_path
    log_prefix = args.log_prefix

    print(f'Random seed: {random_seed}')
    torch.manual_seed(random_seed)

    entrez_p_file = args.summary_data_file

    original_entrez_embedding_file = args.embedding_file

    model = GWAS_literature_MLE(args.fake_data_size, args.embedding_size, hidden_dim, train_time,
                                phi_learning_rate, gwas_threshold,
                                args.gradient_accumulation,
                                save_log,
                                log_save_path,
                                log_prefix,
                                batch_size,
                                multi_hidden=args.multi_hidden)

    if args.use_fake_data:
        fake_data = model.fake_data_generator()

        fg_list, pg_list = model.MLE_for_update(fake_data)
        plot_save_file = f'{self.log_save_path}/{self.log_prefix}.pg-fg.png'
        # save pg-fg scatter.
        model.save_fg_pg_plot(pg_list, fg_list, plot_save_file)
    else:
        print('Use real data.')
        # original_entrez_embedding_file
        print('Training with Real data.')
        entrez_data, entrez_to_p, entrez_to_data = model.load_data(entrez_p_file, original_entrez_embedding_file)

        model.MLE_for_update(entrez_data, entrez_to_p,
                                                args.smoothing_scope, args.smoothing_training_data_size,
                                                args.use_fake_data,
                                                entrez_to_data,
                                                True,
                                                False,
                                                True)


if __name__ == '__main__':
    main()
