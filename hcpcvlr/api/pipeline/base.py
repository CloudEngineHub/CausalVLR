import os
from abc import abstractmethod
import time

import numpy as np
import torch
import pandas as pd
from utils.monitor import Monitor


class BasePipeline(object):
    def __init__(self, model, cfgs):
        self.cfgs = cfgs

        # setup GPU device if available, move model into configured device
        # TODO multi-gpu
        self.model = model.cuda() if torch.cuda.is_available() else model
        
        self.optimizer = None
        self.criterion = None
        self.metric_caculator = None

        # monitor
        self.monitor = Monitor(cfgs)

        self.checkpoint_dir = self.cfgs["result_dir"]
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.start_epoch = 1
        if self.cfgs["resume"] != "":
            self._resume_checkpoint(self.cfgs["resume"])


    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        for epoch in range(self.start_epoch, self.cfgs['epochs'] + 1):
            self._train_epoch(epoch)

            # save logged informations into log dict
            self.monitor.record_best()

            # print logged informations to the screen
            for key, value in self.monitor.name2val.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            is_best, early_stop = self.monitor.check_best()
            if early_stop:
                break

            if epoch % self.cfgs['save_period'] == 0:
                self._save_checkpoint(epoch, save_best=is_best)
            
            self.monitor.dump(epoch)

        self.monitor.print_best_and_save_to_file()
    
    def inference(self, loader):
        pass

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict() if self.optimizer is not None else self.optimizer,
            'monitor_best': self.monitor.monitor_best,
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
        self.monitor.monitor_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        
        assert self.optimizer is not None
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
