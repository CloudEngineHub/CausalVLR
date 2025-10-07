import os
import time
import torch
import json
import pandas as pd
from numpy import inf
from causalvlr.metrics.mrg_metric.metrics import compute_scores
from causalvlr.data.MRG import R2DataLoader
from causalvlr.utils.MRG import tokenizers_fn, build_optimizer, build_lr_scheduler, loss_fn, Monitor
from causalvlr.models.MRG import model_fn


class MRGPipeline(object):
    def __init__(self, args):
        self.args = args
        torch.cuda.set_device(int(args["cuda"]))
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        self.tokenizer = tokenizers_fn[args['tokenizer']](args)
        
        self.train_dataloader = R2DataLoader(args, self.tokenizer, split='train', shuffle=True)
        self.val_dataloader = R2DataLoader(args, self.tokenizer, split='val', shuffle=False)
        self.test_dataloader = R2DataLoader(args, self.tokenizer, split='test', shuffle=False)
        
        self.model = model_fn[args["model"]](args, self.tokenizer)
        self.model = self.model.cuda()
        
        self.criterion = loss_fn[args["loss_fn"]]
        self.metric_ftns = compute_scores
        
        self.optimizer = build_optimizer(args, self.model)
        self.lr_scheduler = build_lr_scheduler(args, self.optimizer, len(self.train_dataloader))
        
        self.epochs = self.args["epochs"]
        self.save_period = self.args["save_period"]
        
        self.mnt_mode = args["monitor_mode"]
        self.mnt_metric = 'val_' + args["monitor_metric"]
        self.mnt_metric_test = 'test_' + args["monitor_metric"]
        assert self.mnt_mode in ['min', 'max']
        
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = self.args["early_stop"]
        
        self.start_epoch = 1
        self.checkpoint_dir = args["result_dir"]
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        if args["resume"] != "":
            self._resume_checkpoint(args["resume"])
        
        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}
        
        self.monitor = Monitor(args)

    def _train_epoch(self, epoch):
        train_loss = 0
        self.model.train()
        start_time = time.time()
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
            images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
            output = self.model(images, reports_ids, mode='train')
            nll_loss = self.criterion(output, reports_ids, reports_masks)
            loss = nll_loss
            self.optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()

            print(f"\repoch: {epoch} {batch_idx}/{len(self.train_dataloader)}\tloss: {loss:.3f}\tmean loss: {train_loss/(batch_idx+1):.3f}",
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
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
                print(f"\rVal Processing: [{int((batch_idx + 1) / len(self.val_dataloader) * 100)}%]", end='',
                      flush=True)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            for k, v in val_met.items():
                self.monitor.logkv(key='val_' + k, val=v)
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                print(f"\rTest Processing: [{int((batch_idx + 1) / len(self.test_dataloader) * 100)}%]", end='',
                      flush=True)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})

            for k, v in test_met.items():
                self.monitor.logkv(key='test_' + k, val=v)
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        if self.args['monitor_metric_curves']:
            self.monitor.plot_current_metrics(epoch, self.monitor.name2val)
        self.monitor.dumpkv(epoch)
        return log

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            if result is None:
                self._save_checkpoint(epoch)
                continue

            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            best = False
            if self.mnt_mode != 'off':
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()

    def inference(self):
        state = torch.load(self.args["load_model_path"], map_location='cuda')
        pretrained_dict = state['state_dict']
        self.model.load_state_dict(pretrained_dict, False)

        log = {'task_name': self.args['task_name']}
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
                print(f"\rVal Processing: [{int((batch_idx + 1) / len(self.val_dataloader) * 100)}%]", end='',
                      flush=True)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                print(f"\rTest Processing: [{int((batch_idx + 1) / len(self.test_dataloader) * 100)}%]", end='',
                      flush=True)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})

            log.update(**{'test_' + k: v for k, v in test_met.items()})

        record_path = os.path.join(self.args["record_dir"], self.args["dataset_name"] + '.csv')
        record_table = pd.DataFrame([log])
        record_table.to_csv(record_path, index=False)
        print(log)
        return log

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args["seed"]
        self.best_recorder['test']['seed'] = self.args["seed"]
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args["record_dir"]):
            os.makedirs(self.args["record_dir"])
        record_path = os.path.join(self.args["record_dir"], self.args["dataset_name"] + '.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = pd.concat([record_table, pd.DataFrame([self.best_recorder['val']]), pd.DataFrame([self.best_recorder['test']])], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'seed': self.args['seed']
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
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args["monitor_metric"]))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args["monitor_metric"]))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))
