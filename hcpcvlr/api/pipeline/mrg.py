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

    def _train_epoch(self, epoch):
        self.model.train()
        start_time = time.time()
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
            images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
            output = self.model(images, reports_ids, mode='train')
            loss = self.criterion(output, reports_ids, reports_masks)
            self.optimizer.zero_grad()
            loss.backward()
            self.monitor.log_mean(('train loss', loss.item()))
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()

            self.monitor.print(f"\repoch: {epoch} {batch_idx+1}/{len(self.train_dataloader)}\tloss: {loss:.3f}\tmean loss: {self.monitor.name2val['train loss']:.3f}", flush=True)

            if self.cfgs["lr_scheduler"] != 'StepLR':
                self.lr_scheduler.step()
        if self.cfgs["lr_scheduler"] == 'StepLR':
            self.lr_scheduler.step()

        self.monitor.print(f"\nEpoch {epoch}\ttime: {time.time() - start_time:.1f}s")

        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            temp_report_ids = []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
                temp_report_ids.extend(output.cpu().numpy())
                self.monitor.print(f"\rVal Processing: [{int((batch_idx + 1) / len(self.val_dataloader) * 100)}%]", flush=True)
            _, n_reports = count_report_pattern(temp_report_ids)
            val_met = self.metric_caculator.compute_scores({i: [gt] for i, gt in enumerate(val_gts)},
                                                           {i: [re] for i, re in enumerate(val_res)})
            # record val metrics
            val_met['n_reports'] = n_reports
            for k, v in val_met.items():
                self.monitor.log(('val_' + k, v))

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res, p = [], [], []
            temp_report_ids = []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                temp_report_ids.extend(output.cpu().numpy())
                self.monitor.print(f"\rTest Processing: [{int((batch_idx + 1) / len(self.test_dataloader) * 100)}%]", flush=True)
            _, n_reports = count_report_pattern(temp_report_ids)
            test_met = self.metric_caculator.compute_scores({i: [gt] for i, gt in enumerate(test_gts)},
                                                            {i: [re] for i, re in enumerate(test_res)})

            test_met['n_reports'] = n_reports
            for k, v in test_met.items():
                self.monitor.log(('test_' + k, v))

        if self.cfgs['monitor_metric_curves']:
            self.monitor.plot_current_metrics(epoch, self.monitor.name2val)


    def inference(self, loader):
        """
        This function takes an integer and a string as input and returns a boolean.
        
        Args:
            loader: 
        
        Returns:
            dict: the score of each metric
        """
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res, report_pattern, img_ids = [], [], [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(loader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())

                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                self.monitor.print(f"\rInference Processing: [{int((batch_idx + 1) / len(loader) * 100)}%]", flush=True)
            # test_res = torch.load("results/mimic_cxr/DMIRG/DMIRG/118_report_100.npy")
            test_met = self.metric_caculator.compute_scores({i: [gt] for i, gt in enumerate(test_gts)},
                                                            {i: [re] for i, re in enumerate(test_res)})
            
            # save_report(test_res, test_gts, img_ids, os.path.join(self.checkpoint_dir, 'report.csv'))
            self.monitor.print(f"\n{test_met}")
            return test_met

def count_report_pattern(report_pattern):
    report_pattern = np.array(report_pattern)
    unique_report_pattern = np.unique(report_pattern, axis=0)
    num_report_pattern = unique_report_pattern.shape[0]
    return unique_report_pattern, num_report_pattern

