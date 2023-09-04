import sys
import os.path as osp
sys.path.append(osp.join(osp.dirname(__file__), '../hcpcvlr'))

from utils.cfgs_loader import load_yaml
from models.mrg.baseline import Baseline
from modules.tokenizers import MRGTokenizer
from api.pipeline import MRGPipeline
from utils.metrics import MetricCalculator
from data import MRGDataLoader
from modules.losses.nlg import compute_lm_loss
from utils.optimizer import build_lr_scheduler, build_optimizer


cfgs = load_yaml("configs/mrg/baseline.yaml")
# print(cfgs)
token = MRGTokenizer(cfgs)
model = Baseline(cfgs, token)

# -------------------
# inference
# -------------------
work = MRGPipeline(model, cfgs, metric_caculator=MetricCalculator(cfgs))
test_dataloader = MRGDataLoader(cfgs, token, split='test', shuffle=False)
work.inference(test_dataloader)
# -------------------
# training
# -------------------
train_dataloader = MRGDataLoader(cfgs, token, split='train', shuffle=True)
val_dataloader = MRGDataLoader(cfgs, token, split='val', shuffle=False)

optimizer = build_optimizer(cfgs, model)
lr_scheduler = build_lr_scheduler(cfgs, optimizer, len(train_dataloader))

work = MRGPipeline(model, cfgs, 
                criterion=compute_lm_loss, 
                metric_caculator=MetricCalculator(cfgs),
                optimizer=lr_scheduler,
                lr_scheduler=optimizer,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader
                )
work.train()
