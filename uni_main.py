"""
Created on 22 May, 2025

@author: TAN Weile
"""

import os
import sys
import argparse
from functools import partial

import time
import numpy as np
from tqdm import tqdm

# debug mode
debug = False
if debug:
    torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='DIDN', help='model name')
parser.add_argument('--dataset_path', default='datasets/diginetica/', help='dataset directory path')
parser.add_argument('--loss_type', default='ce', help='loss type: bce, ce, ce_autocl, bpr-max')
parser.add_argument('--cuda_device', type=int, default=0, help='the id of cuda device')

parser.add_argument('--epoch', type=int, default=120, help='the number of epochs to train')
parser.add_argument('--batch_size', type=int, default=512, help='input batch size 512')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=80,
                    help='the number of steps after which the learning rate decay')
parser.add_argument('--topk', nargs='+', type=int, default=[10, 20], help='a list of top k score items selected for calculating recall and mrr metrics')

parser.add_argument('--single_target', default=False, help='single target')
parser.add_argument('--cl', default=False, help='curriculum learning')
parser.add_argument('--st_mt', default=False, help='get both st_labels and mt_labels from the dataset, only for the experiments about gradient magnitude')
parser.add_argument('--normalize_emb', default=False, help='normalize item and session embeddings when calculating scores')
parser.add_argument('--n_softmaxes', type=int, default=1, help='number of softmaxes, if set to >1, the loss type should be ce')
parser.add_argument('--use_multimax', default=False, help='use multimax loss, if set to True, the loss type should be ce')

# MiaSRec
parser.add_argument('--beta_logit', type=float, default=0.9, help='beta logit')

args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import metric
from dataset import *
from loss import *

# import necessary libraries corresponding to the model
if args.model == 'DIDN':
    import random
    import pickle
    
    from os.path import join
    
    from torch.autograd import Variable
    from torch.backends import cudnn

    from DIDN.didn import DIDN


elif args.model == 'MiaSRec':
    from logging import getLogger

    sys.path.append("MiaSRec")
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_trainer, init_seed, set_color

    from miasrec import MIASREC

    from IPython import embed

elif args.model == 'SASRec':
    sys.path.append("SASRec")
    from SASRec.model import SelfAttentiveSessionEncoder

def trainForEpoch(train_loader, model, optimizer, epoch, num_epochs, criterion, log_aggr=1):
    model.train()

    sum_epoch_loss = 0

    start = time.time()
    if not args.st_mt:
        for i, (given_session, labels, given_session_len, session_index) in tqdm(enumerate(train_loader), total=len(train_loader)):
            given_session = given_session.to(device)

            optimizer.zero_grad()

            if args.model == 'DIDN':
                outputs = model.forward(given_session, given_session_len, normalize_emb=args.normalize_emb)
            elif args.model == 'MiaSRec':
                outputs = model.get_logits(given_session)
            elif args.model == 'SASRec':
                outputs = model.get_logits(given_session)

            if isinstance(criterion, bce_loss_mt):
                loss = criterion(outputs, labels, given_session)
            elif isinstance(criterion, ce_loss_mt):
                loss = criterion(outputs, labels, given_session_len, epoch)
            elif isinstance(criterion, ce_loss_mt_multimax):
                loss = criterion(outputs, labels)
            elif isinstance(criterion, ce_loss_mt_MoS):
                loss = criterion(outputs, labels)
            elif isinstance(criterion, ce_loss_mt_autocl):
                loss = criterion(outputs, labels, given_session_len, epoch)
            elif isinstance(criterion, bpr_max_loss_mt):
                loss = criterion(outputs, labels, epoch)
            else:
                raise ValueError(f"Unknown loss type: {type(criterion)}")

            
            if debug:
                with torch.autograd.detect_anomaly():
                    loss.backward()
            else:
                loss.backward()
            optimizer.step()
            

            loss_val = loss.item()
            sum_epoch_loss += loss_val

            iter_num = epoch * len(train_loader) + i + 1

            if i % log_aggr == 0:
                print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
                    % (epoch + 1, num_epochs, loss_val, sum_epoch_loss / (i + 1),
                        len(given_session) / (time.time() - start)))
                

            start = time.time()

    else:
        avg_gradient_magnitude_st_list = []
        avg_gradient_magnitude_mt_list = []
        for i, (given_session, st_labels, mt_labels, given_session_len, session_index) in tqdm(enumerate(train_loader), total=len(train_loader)):
            given_session = given_session.to(device)

            optimizer.zero_grad()

            outputs = model.forward(given_session, given_session_len, normalize_emb=args.normalize_emb)

            if args.single_target:
                loss = criterion(outputs, st_labels, given_session_len, epoch)
            else:
                loss = criterion(outputs, mt_labels, given_session_len, epoch)

            avg_gradient_magnitude_st, avg_gradient_magnitude_mt = gradient_magnitude(outputs, st_labels, mt_labels, criterion.num_classes, device, args.single_target)
            avg_gradient_magnitude_st_list.append(avg_gradient_magnitude_st.item())
            avg_gradient_magnitude_mt_list.append(avg_gradient_magnitude_mt.item())

            if debug:
                with torch.autograd.detect_anomaly():
                    loss.backward()
            else:
                loss.backward()
            optimizer.step()

            loss_val = loss.item()
            sum_epoch_loss += loss_val

            iter_num = epoch * len(train_loader) + i + 1
            
            if i % log_aggr == 0:
                print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
                    % (epoch + 1, num_epochs, loss_val, sum_epoch_loss / (i + 1),
                        len(given_session) / (time.time() - start)))

            start = time.time()
        
        avg_gradient_magnitude_st = np.mean(avg_gradient_magnitude_st_list)
        avg_gradient_magnitude_mt = np.mean(avg_gradient_magnitude_mt_list)
        print(f"Average gradient magnitude for single-target sessions: {avg_gradient_magnitude_st}")
        print(f"Average gradient magnitude for multi-target sessions: {avg_gradient_magnitude_mt}")

    if hasattr(criterion, 'parameters') and list(criterion.parameters()):
        print("Criterion parameters: ", list(criterion.parameters()))

def validate(valid_loader, model, criterion = None):
    model.eval()
    recalls_list = []
    mrrs_list = []
    with torch.no_grad():
        output_max_list = []
        output_min_list = []
        for given_session, given_session_label, given_session_len in tqdm(valid_loader):
            given_session = given_session.to(device)
            given_session_label = given_session_label.to(device)

            if args.model == 'DIDN':
                outputs = model.forward(given_session, given_session_len, normalize_emb=args.normalize_emb)
            elif args.model == 'MiaSRec':
                outputs = model.get_logits(given_session)
            elif args.model == 'SASRec':
                outputs = model.get_logits(given_session)

            if args.n_softmaxes > 1:
                probs = outputs
            elif isinstance(criterion, ce_loss_mt_multimax):
                # print the range of the outputs
                output_min = outputs.min()
                output_max = outputs.max()
                output_max_list.append(output_max.item())
                output_min_list.append(output_min.item())

                # outputs_flat = outputs.view(-1, 1)
                # logits_flat = criterion.linear_1(outputs_flat)
                # logits_flat = criterion.relu(logits_flat)
                # logits_flat = criterion.linear_2(logits_flat)
                # logits = logits_flat.view(outputs.shape[0], outputs.shape[1])
                # outputs = logits + outputs

                # outputs = SeLU(outputs, criterion.ranges, criterion.ts)

                probs = torch.softmax(outputs, dim=-1)
            else:
                probs = F.softmax(outputs, dim=1)

            recalls, mrrs = metric.evaluate(probs, given_session_label, ks=args.topk)
            recalls_list.append(recalls)
            mrrs_list.append(mrrs)

    mean_recalls = np.mean(recalls_list, axis=0)
    mean_mrrs = np.mean(mrrs_list, axis=0)
    if output_max_list:
        output_max = np.max(output_max_list)
        output_min = np.min(output_min_list)
        print(f"Output range: {output_min} to {output_max}")
    return mean_recalls, mean_mrrs
    

if __name__ == '__main__':
    dataset_name = args.dataset_path.split('/')[-2]
    if dataset_name in ['diginetica', 'yoochoose4', 'yoochoose64', 'lastfm', 'gowalla', 'retailrocket']:
        with open(args.dataset_path + 'num_items.txt', 'r') as f:
            n_items = int(f.readline().strip())
    else:
        raise Exception(f'Unknown Dataset! {dataset_name}')
    
    if args.model == 'DIDN':

        max_len = 19

        model = DIDN(n_items=n_items,
                     hidden_size=64,
                     embedding_dim=64,
                     batch_size=args.batch_size,
                     max_len=max_len,
                     position_embed_dim=64,
                     alpha1=0.1,
                     alpha2=0.1,
                     alpha3=0.1,
                     pos_num=2000,
                     neighbor_num=5,
                     n_softmaxes=args.n_softmaxes)

        padding_direction = 'right'
        
    elif args.model == 'MiaSRec':        
        config = {}
        config['n_layers'] = 2
        config['n_heads'] = 2
        config['hidden_size'] = 100
        config['inner_size'] = 256
        config['hidden_dropout_prob'] = 0.1
        config['attn_dropout_prob'] = 0.1
        config['hidden_act'] = 'gelu'
        config['layer_norm_eps'] = 1e-12
        config['initializer_range'] = 0.02
        config['entmax_alpha'] = -1
        config['max_repeat'] = 2
        config['seqlen'] = 50
        max_len = 50

        config['beta_logit'] = args.beta_logit

        config['n_items'] = n_items
        config['embedding_size'] = 100
        config['device'] = device
        config['sess_dropout'] = 0.1
        config['item_dropout'] = 0.1
        config['temperature'] = 0.07

        model = MIASREC(config)

        padding_direction = 'right'

    elif args.model == 'SASRec':

        max_len = 19

        model = SelfAttentiveSessionEncoder(num_items=n_items,
                                            hidden_size=64,
                                            n_layers=2,
                                            n_head=1,
                                            max_session_length=max_len,
                                            hidden_dropout_prob=0.2)
        
        padding_direction = 'left'

    else:
        raise ValueError(f'Model {args.model} not found')
    
    print(model.parameters)
    model.to(device)

    print("Loading data...")
    train, valid, test = load_retrieved_data(args.dataset_path)
    train_data = RecSysDatasetTrain(train)
    valid_data = RecSysDatasetValid(valid)
    test_data = RecSysDatasetValid(test)
    if args.st_mt:
        train_loader = DataLoader(train_data, 
                                  batch_size=args.batch_size, 
                                  shuffle=True, 
                                  collate_fn=partial(collate_fn_train_st_mt, max_session_len=max_len, padding_direction=padding_direction))
    elif args.single_target:
        train_loader = DataLoader(train_data, 
                                  batch_size=args.batch_size, 
                                  shuffle=True, 
                                  collate_fn=partial(collate_fn_train, max_session_len=max_len, single_target=True, padding_direction=padding_direction))
    else:
        train_loader = DataLoader(train_data, 
                                batch_size=args.batch_size, 
                                shuffle=True, 
                                collate_fn=partial(collate_fn_train, max_session_len=max_len, padding_direction=padding_direction))
    valid_loader = DataLoader(valid_data, 
                              batch_size=args.batch_size, 
                              shuffle=False, 
                              collate_fn=partial(collate_fn_valid, max_session_len=max_len, padding_direction=padding_direction))
    test_loader = DataLoader(test_data, 
                             batch_size=args.batch_size, 
                             shuffle=False, 
                             collate_fn=partial(collate_fn_valid, max_session_len=max_len, padding_direction=padding_direction))
    
    if args.n_softmaxes > 1:
        if args.loss_type != 'ce':
            raise ValueError(f"Loss type {args.loss_type} is not supported for MoS")
        criterion = ce_loss_mt_MoS(num_classes=n_items, device=device)
    elif args.use_multimax:
        criterion = ce_loss_mt_multimax(num_classes=n_items, device=device)
    elif args.loss_type == 'bce':
        criterion = bce_loss_mt(num_classes=n_items, device=device)
    elif args.loss_type == 'ce':
        criterion = ce_loss_mt(num_classes=n_items, device=device, cl = args.cl)
    elif args.loss_type == 'ce_autocl':
        criterion = ce_loss_mt_autocl(num_classes=n_items, device=device)
    elif args.loss_type == 'bpr-max':
        criterion = bpr_max_loss_mt(num_classes=n_items, device=device)
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")
    
    if hasattr(criterion, 'parameters') and list(criterion.parameters()):
        print("Criterion has parameters: ", list(criterion.parameters()))
        optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), args.lr)

    scheduler = StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

    best_recall = [0.0] * len(args.topk)
    best_mrr = [0.0] * len(args.topk)

    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    best_model_path = f'best_model_weight/best_model_{timestamp}.pth'
    print(f"Best model path: {best_model_path}")

    for epoch in tqdm(range(args.epoch)):
        # train for one epoch
        trainForEpoch(train_loader, model, optimizer, epoch, args.epoch, criterion, log_aggr=200)
        scheduler.step(epoch=epoch)


        recalls, mrrs = validate(valid_loader, model, criterion)
        
        log_str = 'Epoch {} validation: '.format(epoch)
        for k, r, m in zip(args.topk, recalls, mrrs):
            log_str += 'Recall@{}: {:.4f}, MRR@{}: {:.4f} | '.format(k, r, k, m)
        print(log_str)
        
        # save the best model
        if recalls[-1] > best_recall[-1]:
            best_recall = recalls
            best_mrr = mrrs
            
            torch.save(model.state_dict(), best_model_path)
            print("Best model saved.")

    log_str = "Best model on validation set: "
    for k, r, m in zip(args.topk, best_recall, best_mrr):
        log_str += "Recall@{}: {:.4f}, MRR@{}: {:.4f} | ".format(k, r * 100, k, m * 100)
    print(log_str)

    # load the best model and evaluate on the test set
    model.load_state_dict(torch.load(best_model_path))
    recalls, mrrs = validate(test_loader, model, criterion)
    log_str = "Test results: "
    for k, r, m in zip(args.topk, recalls, mrrs):
        log_str += "Recall@{}: {:.4f}, MRR@{}: {:.4f} | ".format(k, r * 100, k, m * 100)
    print(log_str)


