import os
from abc import abstractmethod
import time

import numpy as np
import torch
import pandas as pd
from numpy import inf


class BasePipeline(object):
    def __init__(self, model, cfgs):
        self.cfgs = cfgs

        # setup GPU device if available, move model into configured device
        # TODO multi-gpu
        self.model = model.cuda() if torch.cuda.is_available() else model
        
        self.optimizer = None
        self.criterion = None
        self.metric_caculator = None


        self.monitor_mode = cfgs["monitor_mode"]
        self.monitor_metric = 'val_' + self.cfgs["monitor_metric"]
        self.monitor_metric_test = 'test_' + self.cfgs["monitor_metric"]
        assert self.monitor_mode in ['min', 'max']

        self.monitor_best = inf if self.monitor_mode == 'min' else -inf
        self.checkpoint_dir = self.cfgs["result_dir"]

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if self.cfgs["resume"] != "":
            self._resume_checkpoint(self.cfgs["resume"])

        self.best_recorder = {'val': {self.monitor_metric: self.monitor_best},
                              'test': {self.monitor_metric_test: self.monitor_best}}


    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
    
    def inference(self, loader=None):
        pass

    def _print_best_to_file(self):
        # TODO
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.cfgs["seed"]
        self.best_recorder['test']['seed'] = self.cfgs["seed"]
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.cfgs["record_dir"]):
            os.makedirs(self.cfgs["record_dir"])
        record_path = os.path.join(self.cfgs["record_dir"], self.cfgs["dataset_name"] + '.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict() if self.optimizer is not None else self.optimizer,
            'monitor_best': self.monitor_best,
            'seed': self.cfgs['seed']
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("*************** Saving current best: model_best.pth ... ***************")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        
        assert self.optimizer is not None
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.monitor_mode == 'min' and log[self.monitor_metric] <= self.best_recorder['val'][
            self.monitor_metric]) or \
                       (self.monitor_mode == 'max' and log[self.monitor_metric] >= self.best_recorder['val'][self.monitor_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.monitor_mode == 'min' and log[self.monitor_metric_test] <= self.best_recorder['test'][
            self.monitor_metric_test]) or \
                        (self.monitor_mode == 'max' and log[self.monitor_metric_test] >= self.best_recorder['test'][
                            self.monitor_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.cfgs["monitor_metric"]))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.cfgs["monitor_metric"]))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))
