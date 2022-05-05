from __future__ import print_function
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.nn.functional as F
import os
import torchnet as tnt
import utils
import pickle
from tqdm import tqdm
import time
import numpy as np

from random import shuffle,sample
from . import Algorithm
from pdb import set_trace as breakpoint


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        if batch_size == 0:
            res.append(correct_k.mul_(0))
        else:
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

class Model(Algorithm):
    def __init__(self, opt):
        Algorithm.__init__(self, opt)
        self.temperature = self.opt['data_train_opt']['temperature']              
        self.alpha = self.opt['data_train_opt']['alpha']
        self.w_inst = self.opt['data_train_opt']['w_inst']
        self.w_recon = self.opt['data_train_opt']['w_recon']
        
    def train_step(self, batch, warmup):       
        if self.opt['knn'] and self.curr_epoch>=self.opt['knn_start_epoch'] and not warmup:
            return self.train_pseudo(batch)
        else:   
            return self.train(batch,warmup=warmup)

    def train_naive(self, batch):
        data = batch[0].cuda(non_blocking=True)
        target = batch[1].cuda(non_blocking=True)
        record = {}
        
        output,_ = self.networks['model'](data)
        loss = self.criterions['loss'](output,target)
        record['loss'] = loss.item()
        record['train_accuracy'] = accuracy(output,target)[0].item()  
        
        self.optimizers['model'].zero_grad()
        loss.backward()            
        self.optimizers['model'].step()          
        return record    
    
    
    def train(self, batch, warmup=True):
        data = batch[0].cuda(non_blocking=True)
        target = batch[1].cuda(non_blocking=True)
        batch_size = data.size(0)
        record = {}
        
        output,feat = self.networks['model'](data,do_recon=False)
        loss = self.criterions['loss'](output,target)
        record['loss'] = loss.item()
        record['train_accuracy'] = accuracy(output,target)[0].item()   

        if not warmup:   
            data_aug = batch[3].cuda(non_blocking=True)       

            shuffle_idx = torch.randperm(batch_size)
            mapping = {k:v for (v,k) in enumerate(shuffle_idx)}
            reverse_idx = torch.LongTensor([mapping[k] for k in sorted(mapping.keys())])     
            feat_aug = self.networks['model'](data_aug[shuffle_idx],do_recon=False,has_out=False)  
            feat_aug = feat_aug[reverse_idx]
            
            ##**************Instance contrastive loss****************
            sim_clean = torch.mm(feat, feat.t())
            mask = (torch.ones_like(sim_clean) - torch.eye(batch_size, device=sim_clean.device)).bool()
            sim_clean = sim_clean.masked_select(mask).view(batch_size, -1)

            sim_aug = torch.mm(feat, feat_aug.t())
            sim_aug = sim_aug.masked_select(mask).view(batch_size, -1)   
            
            logits_pos = torch.bmm(feat.view(batch_size,1,-1),feat_aug.view(batch_size,-1,1)).squeeze(-1)
            logits_neg = torch.cat([sim_clean,sim_aug],dim=1)

            logits = torch.cat([logits_pos,logits_neg],dim=1)
            instance_labels = torch.zeros(batch_size).long().cuda()
            
            loss_instance = self.criterions['loss_instance'](logits/self.temperature, instance_labels)                
            loss += self.w_inst*loss_instance
            record['loss_inst'] = loss_instance.item()
            record['acc_inst'] = accuracy(logits,instance_labels)[0].item()           
            
            # ##**************Mixup Prototypical contrastive loss****************     
            #'''
            L = np.random.beta(self.alpha, self.alpha)     
            labels = torch.zeros(batch_size, self.opt['data_train_opt']['num_class']).cuda().scatter_(1, target.view(-1,1), 1) 
            
            if 'cifar' in self.opt['dataset']:
                inputs = torch.cat([data,data_aug],dim=0)
                idx = torch.randperm(batch_size*2) 
                labels = torch.cat([labels,labels],dim=0)
            else: #do not use augmented data to save gpu memory    
                inputs = data            
                idx = torch.randperm(batch_size)              
            
            #input_mix = L * inputs + (1 - L) * inputs[idx]  
            #labels_mix = L * labels + (1 - L) * labels[idx]
            input_mix = inputs
            labels_mix = labels
               
            feat_mix = self.networks['model'](input_mix,has_out=False)  

            logits_proto = torch.mm(feat_mix,self.prototypes.t())/self.temperature      
            loss_proto = -torch.mean(torch.sum(F.log_softmax(logits_proto, dim=1) * labels_mix, dim=1))
            record['loss_proto'] = loss_proto.item()          
            loss += self.w_proto*loss_proto  
            #'''

            if False and True:
                # Do Supervised Contrastive Loss
                clean_idx = torch.ones_like(target).bool() 
                clean_flag_new = clean_idx.view(-1, 1)  # (N, 1)
                clean_mask = torch.eq(clean_flag_new, clean_flag_new.T).float() * clean_flag_new.float()  # (N, N)
                tmp_mask = (torch.ones_like(clean_mask) - torch.eye(batch_size, device=clean_mask.device)).bool()
                clean_mask = clean_mask.masked_select(tmp_mask).view(batch_size, -1)  # (N, N-1)
                clean_mask = torch.cat(
                    (torch.ones(batch_size, device=clean_mask.device).view(-1, 1), clean_mask, clean_mask),
                    dim=1)  # (N, 2N-1), clean flag to logits_icl
                gt_label_new = target.view(-1, 1)  # (N, 1)
                inst_labels = torch.eq(gt_label_new, gt_label_new.T).float()  # (N, N)
                inst_mask = (torch.ones_like(inst_labels) - torch.eye(batch_size, device=inst_labels.device)).bool()
                inst_labels = inst_labels.masked_select(inst_mask).view(batch_size, -1).cuda()
                inst_labels = torch.cat(
                    (torch.ones(batch_size, device=inst_labels.device).view(-1, 1), inst_labels, inst_labels),
                    dim=1)  # (N, 2N-1), labels to logits_icl
                inst_labels = inst_labels * clean_mask.cuda()  # (N, 2N-1), only use the clean instances
                log_prob = logits - torch.log(torch.exp(logits).sum(1, keepdim=True))
                mean_log_prob_pos = (inst_labels * log_prob).sum(1) / inst_labels.sum(1)
                #loss_sup = -1 * (weights * mean_log_prob_pos.view(1, batch_size)).sum() / weights.sum()
                loss_sup = -1 * mean_log_prob_pos.view(1, batch_size).mean()
                record['loss_scl'] = loss_sup.item()
                loss += self.w_proto*loss_sup
            
        self.optimizers['model'].zero_grad()
        loss.backward()            
        self.optimizers['model'].step()       
        
        return record 
    
    
    def train_pseudo(self, batch):
        data = batch[0].cuda(non_blocking=True)   
        data_aug = batch[3].cuda(non_blocking=True)    
        index = batch[2] 
        batch_size = data.size(0)
        target = self.hard_labels[index].cuda(non_blocking=True)
        #soft_target = self.soft_labels[index].cuda(non_blocking=True)
        clean_idx = self.clean_idx[index] 
        weights = self.weights[index].cuda(non_blocking=True)
        # weights = self.loss_weights[index].cuda(non_blocking=True)

        record = {}
        
        output,feat = self.networks['model'](data,do_recon=False)

        #L = np.random.beta(self.alpha, self.alpha)     
        #inputs = data         
        #idx = torch.randperm(len(inputs))  
        #labels = torch.zeros(batch_size, self.opt['data_train_opt']['num_class']).cuda().scatter_(1, target.view(-1,1), 1)
        #input_mix = L * inputs + (1 - L) * inputs[idx]
        #labels_mix = L * labels + (1 - L) * labels[idx]
        #sample_weights = L * weights + (1 - L) * weights[idx]
        
        loss = torch.sum(weights[clean_idx] * nn.CrossEntropyLoss(reduction='none')(output[clean_idx], target[clean_idx])) / torch.sum(weights[clean_idx])
        #loss = torch.mean(self.criterions['loss'](output[clean_idx],target[clean_idx]))

        #loss = -torch.sum(weights * torch.sum(F.log_softmax(output, dim=1) * soft_target, dim=1)) / torch.sum(weights)

        record['loss'] = loss.item()
        record['train_accuracy'] = accuracy(output[clean_idx],target[clean_idx])[0].item()          

        shuffle_idx = torch.randperm(batch_size)
        mapping = {k:v for (v,k) in enumerate(shuffle_idx)}
        reverse_idx = torch.LongTensor([mapping[k] for k in sorted(mapping.keys())])     
        feat_aug = self.networks['model'](data_aug[shuffle_idx],do_recon=False,has_out=False)   
        feat_aug = feat_aug[reverse_idx]
              

        ##**************Instance contrastive loss****************
        sim_clean = torch.mm(feat, feat.t())
        mask = (torch.ones_like(sim_clean) - torch.eye(batch_size, device=sim_clean.device)).bool()
        sim_clean = sim_clean.masked_select(mask).view(batch_size, -1)

        sim_aug = torch.mm(feat, feat_aug.t())
        sim_aug = sim_aug.masked_select(mask).view(batch_size, -1)   

        logits_pos = torch.bmm(feat.view(batch_size,1,-1),feat_aug.view(batch_size,-1,1)).squeeze(-1)
        logits_neg = torch.cat([sim_clean,sim_aug],dim=1)

        logits = torch.cat([logits_pos,logits_neg],dim=1)
        instance_labels = torch.zeros(batch_size).long().cuda()

        loss_instance = self.criterions['loss_instance'](logits/self.temperature, instance_labels)                
        loss += self.w_inst*loss_instance
        record['loss_inst'] = loss_instance.item()
        record['acc_inst'] = accuracy(logits,instance_labels)[0].item()           

        ##**************Mixup Prototypical contrastive loss****************     

        if True and sum(clean_idx) > 0:
            L = np.random.beta(self.alpha, self.alpha)    
            
            labels = torch.zeros(batch_size, self.opt['data_train_opt']['num_class']).cuda().scatter_(1, target.view(-1,1), 1)  
            #labels = soft_target

            if 'cifar' in self.opt['dataset']:
                inputs = torch.cat([data[clean_idx],data_aug[clean_idx]],dim=0)
                idx = torch.randperm(clean_idx.sum()*2) 
                labels = torch.cat([labels[clean_idx],labels[clean_idx]],dim=0)
                weights = torch.cat([weights[clean_idx], weights[clean_idx]], dim=0)
            else: #do not use augmented data to save gpu memory    
                inputs = data[clean_idx]            
                labels = labels[clean_idx]
                idx = torch.randperm(clean_idx.sum())  

            input_mix = L * inputs + (1 - L) * inputs[idx]  
            labels_mix = L * labels + (1 - L) * labels[idx]
            #weights = torch.max(weights, weights[idx])[0]
            weights = L * weights + (1 - L) * weights[idx]

            feat_mix = self.networks['model'](input_mix,has_out=False)  

            logits_proto = torch.mm(feat_mix,self.prototypes.t())/self.temperature

            #logits_proto = torch.mm(feat_mix,self.prototypes.t())
            #print (self.epsilons[target[clean_idx]].view(-1,1))
            #print (type(self.epsilons[target[clean_idx]].view(-1,1)))
            #margin = torch.zeros(logits_proto.shape, dtype=torch.float64).cuda().scatter_(1, labels.view(-1,1), self.epsilons[labels].view(-1,1))
            #logits_proto -= margin
            # loss_proto = -torch.mean(torch.sum(F.log_softmax(logits_proto, dim=1) * labels_mix, dim=1))
            #cls_weights = torch.cat([self.epsilons[labels[clean_idx]], self.epsilons[labels[clean_idx]]], dim=0)
            loss_proto = -torch.sum(weights * torch.sum(F.log_softmax(logits_proto, dim=1) * labels_mix, dim=1)) / torch.sum(weights)
            record['loss_proto'] = loss_proto.item()          
            loss += self.w_proto*loss_proto          

        if False and True:
            # Do Supervised Contrastive Loss
            clean_flag_new = clean_idx.view(-1, 1)  # (N, 1)
            clean_mask = torch.eq(clean_flag_new, clean_flag_new.T).float() * clean_flag_new.float()  # (N, N)
            tmp_mask = (torch.ones_like(clean_mask) - torch.eye(batch_size, device=clean_mask.device)).bool()
            clean_mask = clean_mask.masked_select(tmp_mask).view(batch_size, -1)  # (N, N-1)
            clean_mask = torch.cat(
                (torch.ones(batch_size, device=clean_mask.device).view(-1, 1), clean_mask, clean_mask),
                dim=1)  # (N, 2N-1), clean flag to logits_icl
            gt_label_new = target.view(-1, 1)  # (N, 1)
            inst_labels = torch.eq(gt_label_new, gt_label_new.T).float()  # (N, N)
            inst_mask = (torch.ones_like(inst_labels) - torch.eye(batch_size, device=inst_labels.device)).bool()
            inst_labels = inst_labels.masked_select(inst_mask).view(batch_size, -1).cuda()
            inst_labels = torch.cat(
                (torch.ones(batch_size, device=inst_labels.device).view(-1, 1), inst_labels, inst_labels),
                dim=1)  # (N, 2N-1), labels to logits_icl
            inst_labels = inst_labels * clean_mask.cuda()  # (N, 2N-1), only use the clean instances
            log_prob = logits - torch.log(torch.exp(logits).sum(1, keepdim=True))
            mean_log_prob_pos = (inst_labels * log_prob).sum(1) / inst_labels.sum(1)
            loss_sup = -1 * (weights * mean_log_prob_pos.view(1, batch_size)).sum() / weights.sum()
            #loss_sup = -1 * mean_log_prob_pos.view(1, batch_size).mean()
            record['loss_scl'] = loss_sup.item()
            loss += self.w_proto*loss_sup

        self.optimizers['model'].zero_grad()
        loss.backward()            
        self.optimizers['model'].step()       
        
        return record 
        
