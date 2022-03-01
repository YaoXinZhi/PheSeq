# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 15/10/2021 9:24
@Author: XINZHI YAO
"""

"""
Model 3.1
1. add DME to model.
"""


import os
import math
import logging

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

from module.args import *
from module.DynamicMeataEmbedding_module import get_embedder


class evidence_encoder(torch.nn.Module):
    def __init__(self, args):
        super(evidence_encoder, self).__init__()

        self.embedder = get_embedder(args)

        self.hidden_nn = nn.Linear(args.final_embedding_size, args.hidden_size)

        self.beta_nn = nn.Linear(args.hidden_size, 3)

        self.relu = nn.ReLU()

    def forward(self, entrez):
        hidden_state = self.embedder(entrez)
        # print(hidden_state.shape)
        hidden_state = self.hidden_nn(hidden_state)

        _ag, _bg, _alpha_g = self.relu(self.beta_nn(hidden_state)).split(1, dim=-1)

        # avid zero for ag/bg/alpha_g
        _ag = _ag + 1e-10
        _bg = _bg + 1e-10
        _alpha_g = _alpha_g + 1e-10

        return _ag, _bg, _alpha_g

    def model_params(self):
        return [ *self.hidden_nn.parameters(), *self.beta_nn.parameters() ]

class GWAS_literature_MLE:

    def __init__(self, args):
        self.G = args.entrez_size

        self.significant_threshold = args.significant_threshold
        self.train_time = args.train_time
        self.phi_learning_rate = args.phi_learning_rate
        self.gradient_accumulation = args.gradient_accumulation
        self.batch_size = args.batch_size
        self.random_seed = args.random_seed

        self.save_log = args.save_log
        self.log_save_path = args.log_save_path
        self.log_prefix = args.log_prefix

        self.use_best_alpha = args.use_best_alpha

        # Pythonic
        self.entrez_to_p = {}

        self.T, self.a, self.b, self.alpha, self.P, self.F = None, None, None, None, None, None
        self.objective_function = None
        self.logger = None

        self.alpha_trained = False
        self.phi_trained = False

        self.logger_init()

        self.seed_init()

        self.logger.info('Initializing encoder.')
        self.model = evidence_encoder(args)

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
    def get_best_alpha(entrez_to_p: dict):
        entrez_to_alpha = {}
        for entrez, p in entrez_to_p.items():
            if p == 0:
                p = 1e-128

            alpha_g_best = - (1 / math.log(p))
            entrez_to_alpha[entrez] = alpha_g_best

        return entrez_to_alpha

    def load_data(self, p_value_file: str):
        self.logger.info('loading entrez_p_file.')
        entrez_to_p = self.read_entrez_p(p_value_file)

        entrez_data = []
        for entrez, p in entrez_to_p.items():
            if p == 0.0:
                p = 1e-38

            entrez_data.append((entrez, torch.tensor(p)))

        random.shuffle(entrez_data)
        self.logger.info(f'data size: {len(entrez_data)}')

        return entrez_data, entrez_to_p

    def objective_function_init(self):

        self.T, self.a, self.b, self.alpha, self.P, self.F = sympy.symbols('T a b alpha P F')

        self.objective_function = self.T*sympy.log(self.alpha) \
                 + self.T * (self.alpha - 1) * sympy.log(self.P) \
                 + (self.T+self.a-1)*sympy.log(self.F) \
                 + (self.b-self.T)*sympy.log(1-self.F) \
                 + sympy.log(sympy.gamma(self.a+self.b)/sympy.gamma(self.a)/sympy.gamma(self.b))

    @staticmethod
    def early_stopping(ag_quo_list: list, bg_quo_list: list, alpha_g_quo_list: list):

        ag_quo_con_por = len([ag for ag in ag_quo_list if 0.95 < ag < 1.1 ]) / len(ag_quo_list)
        bg_quo_con_por = len([bg for bg in bg_quo_list if 0.95 < bg < 1.1 ]) / len(bg_quo_list)
        alpha_g_quo_con_por = len([ alpha_g for alpha_g in alpha_g_quo_list if 0.95 < alpha_g < 1.1 ]) / len(alpha_g_quo_list)

        if ag_quo_con_por > 0.9 and bg_quo_con_por > 0.9 and alpha_g_quo_con_por > 0.9:
            return True, ag_quo_con_por, bg_quo_con_por, alpha_g_quo_con_por
        else:
            return False, ag_quo_con_por, bg_quo_con_por, alpha_g_quo_con_por


    def MLE_for_update(self, entrez_data: list, entrez_to_p):

        self.logger.info('Start training for phi and alpha update.')

        if self.use_best_alpha:
            entrez_to_alpha = self.get_best_alpha(entrez_to_p)
        else:
            entrez_to_alpha = None

        ag_t_list = []
        bg_t_list = []
        alpha_g_t_list = []

        ag_quo_list = []
        bg_quo_list = []
        alpha_quo_list = []

        wf_ag_quo = open(f'{self.log_save_path}/ag_quo.txt', 'w')
        wf_bg_quo = open(f'{self.log_save_path}/bg_quo.txt', 'w')
        wf_alpha_quo = open(f'{self.log_save_path}/alpha_g_quo.txt', 'w')

        fg_list = []
        pg_list = []
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
            for g in range(len(entrez_data)):
                if g % int(len(entrez_data) / 10) == 0:
                    logging.info(f'Training:\ttime: {time}, g: {g}/{len(entrez_data)} training done.')

                entrez_g, pg_ture = entrez_data[g]
                if pg_ture == 0:
                    pg_ture = torch.tensor(1e-38)

                # lg = torch.from_numpy(lg).float()

                # single number now
                ag, bg, alpha_g = self.model(entrez_g)

                ag_detach = ag.detach()
                bg_detach = bg.detach()
                alpha_g_detach = alpha_g.detach()

                fg = torch.distributions.Beta(ag_detach, bg_detach).sample()

                Tg = torch.distributions.Bernoulli(fg).sample()

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

                # save fg/pg for Drawing
                fg_list.append(fg.detach().item())
                pg_list.append(pg_ture.item())

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
                            # break
                            if ag_para_grad.shape != bg_para_grad.shape \
                                or ag_para_grad.shape != alpha_g_para_grad.shape:
                                raise ValueError(f'ag_para_grad.shape: {ag_para_grad.shape}, grad.shape: {ag_para_grad.shape}')

                            grad = float(grad_ag) * ag_para_grad \
                                    + float(grad_bg) * bg_para_grad \
                                    + float(grad_alpha_g) * alpha_g_para_grad

                            new_para = para + self.phi_learning_rate * grad
                            # print('single update')
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

            if early_stop_bool:
                self.logger.info(f'Early stopping, time: {time}.')
                self.logger.info(f'ag_convergence: {ag_quo_con_por:.2f}, '
                                 f'bg_convergence: {bg_quo_con_por:.2f}, '
                                 f'alpha_g_convergence: {alpha_g_quo_con_por:.2f}.')

                fg_pg_save_file = f'{self.log_save_path}/fg_pg.tsv'
                with open(fg_pg_save_file, 'w') as wf:
                    for fg, pg in zip(fg_list, pg_list):
                        wf.write(f'{fg}\t{pg}\n')
                break
            elif time > 10:
                self.logger.info(f'Training done, time: {time}.')
                self.logger.info(f'ag_convergence: {ag_quo_con_por:.2f}, '
                                 f'bg_convergence: {bg_quo_con_por:.2f}, '
                                 f'alpha_g_convergence: {alpha_g_quo_con_por:.2f}.')

                fg_pg_save_file = f'{self.log_save_path}/fg_pg.tsv'
                with open(fg_pg_save_file, 'w') as wf:
                    for fg, pg in zip(fg_list, pg_list):
                        wf.write(f'{fg}\t{pg}\n')
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
                        new_para = para + self.phi_learning_rate * para_grad / len(L_G)
                        # print('batch update')
                        para.copy_(new_para)
            self.evaluate(entrez_data, time=time, final_eval=False,
                          entrez_to_alpha=None, use_best_alpha=False)

        self.evaluate(entrez_data, time=time, final_eval=True,
                      entrez_to_alpha=entrez_to_alpha, use_best_alpha=self.use_best_alpha)

        return fg_list, pg_list


    def evaluate(self, eval_data_set: list, time: int, final_eval: bool,
                 entrez_to_alpha=None, use_best_alpha=False):
        # reconstruction error
        self.logger.info('start evaluate.')
        pred_p_list = []
        ture_p_list = []
        t_list = []
        f_list = []

        entrez_list = []

        with torch.no_grad():
            for g in range(len(eval_data_set)):

                entrez_g, pg_ture = eval_data_set[ g ]

                # lg = torch.from_numpy(lg).float()

                ag, bg, alpha_g = self.model(entrez_g)

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

                self.logger.info(f'Evaluation\tDataIdx:{g}\t'
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
            pred_save_file = f'{self.log_save_path}/{self.log_prefix}.p-pred.tsv'

            with open(pred_save_file, 'w') as wf:
                wf.write(f'Entrez\tF_g\tT_g\tTure P-value\tPred P-value\n')
                for g, entrez in enumerate(entrez_list):
                    wf.write(f'{entrez}\t{f_list[g]}\t{t_list[g]}\t{ture_p_list[g]}\t{pred_p_list[g]}\n')

            self.logger.info(f'{pred_save_file} save done.')
            
    def save_fg_pg_plot(self, pg_list: list, fg_list: list):

        self.logger.info(f'Saving pg-fg scatter.')
        plot_save_file = f'{self.log_save_path}/{self.log_prefix}.pg-fg.png'

        fig = plt.figure(figsize=(100, 2))

        x_major_locator = MultipleLocator(0.01)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)

        plt.scatter(pg_list, fg_list)

        fig.savefig(plot_save_file)
        self.logger.info(f'{plot_save_file} save done.')

    def inference(self):
        pass

def main():
    pass


if __name__ == '__main__':
    pass
