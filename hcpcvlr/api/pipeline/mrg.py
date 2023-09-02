"""
pipeline for medical report generation
"""
import pandas as pd
from .base import BasePipeline
import numpy as np
import torch
import time
import os


class MRGPipeline(BasePipeline):
    def __init__(self, model, cfgs, 
            criterion=None, 
            metric_caculator=None, 
            optimizer=None, 
            lr_scheduler=None, 
            train_dataloader=None, 
            val_dataloader=None, 
            test_dataloader=None
    ):
        super(MRGPipeline, self).__init__(model, cfgs)

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.metric_caculator = metric_caculator
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
    
    def train(self):
        for epoch in range(self.cfgs["epoch"]):
            self._train_epoch(epoch)

    def _train_epoch(self, epoch):
        train_loss = 0
        self.model.train()
        start_time = time.time()
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
            images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
            output = self.model(images, reports_ids, mode='train')
            loss = self.criterion(output, reports_ids, reports_masks)
            self.optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()

            print(f"\repoch: {epoch+1} {batch_idx+1}/{len(self.train_dataloader)}\tloss: {loss:.3f}\tmean loss: {train_loss/(batch_idx+1):.3f}",
                  flush=True, end='')

            if self.args["lr_scheduler"] != 'StepLR':
                self.lr_scheduler.step()
        if self.args["lr_scheduler"] == 'StepLR':
            self.lr_scheduler.step()

        log = {'train_loss': train_loss / len(self.train_dataloader)}
        print("\n")
        print("\tEpoch {}\tmean_loss: {:.4f}\ttime: {:.4f}s".format(epoch, log['train_loss'], time.time() - start_time))

        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            p = torch.zeros([1, self.args["max_seq_length"]]).cuda()
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
                p = torch.cat([p, output])
                print(f"\rVal Processing: [{int((batch_idx + 1) / len(self.val_dataloader) * 100)}%]", end='',
                      flush=True)
            tp, lp = count_p(p[1:])
            val_met = self.metric_fns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            # record val metrics
            for k, v in val_met.items():
                self.monitor.logkv(key='val_' + k, val=v)
            val_met['p'] = lp
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res, p = [], [], []
            p = torch.zeros([1, self.args["max_seq_length"]]).cuda()
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                p = torch.cat([p, output])
                print(f"\rTest Processing: [{int((batch_idx + 1) / len(self.test_dataloader) * 100)}%]", end='',
                      flush=True)
            tp, lp = count_p(p[1:])
            test_met = self.metric_fns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})

            for k, v in test_met.items():
                self.monitor.logkv(key='test_' + k, val=v)
            test_met['p'] = lp
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        if self.args['monitor_metric_curves']:
            self.monitor.plot_current_metrics(epoch, self.monitor.name2val)
        self.monitor.dumpkv(epoch)
        return log

    def inference(self, loader=None):
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res, report_pattern, img_ids = [], [], [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())

                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                print(f"\rTest Processing: [{int((batch_idx + 1) / len(self.test_dataloader) * 100)}%]", end='',
                      flush=True)
            # test_res = torch.load("results/mimic_cxr/DMIRG/DMIRG/118_report_100.npy")
            test_met = self.metric_fns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            
            save_report(test_res, test_gts, img_ids, os.path.join(self.checkpoint_dir, 'report.csv'))
            print(test_met)
            # print(lp)

def count_report_pattern(report_pattern):
    t = torch.unique(report_pattern, dim=0)
    l = t.size(0)
    return unique_report_pattern, num_report_pattern

