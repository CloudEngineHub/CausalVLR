from hcpcvlr.utils.cfgs_loader import load_yaml
from hcpcvlr.models.mrg import Baseline, VLCI
from hcpcvlr.modules.tokenizers import MRGTokenizer
from hcpcvlr.api.pipeline import MRGPipeline
from hcpcvlr.utils.metrics import MetricCalculator
from hcpcvlr.data import MRGDataLoader
from hcpcvlr.modules.losses.nlg import compute_lm_loss
from hcpcvlr.utils.optimizer import build_lr_scheduler, build_optimizer

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
cfgs = load_yaml("configs/mrg/vlci.yaml")
token = MRGTokenizer(cfgs)
model = VLCI(cfgs, token)
test_dataloader = MRGDataLoader(cfgs, token, split='test', shuffle=False)
train_dataloader = MRGDataLoader(cfgs, token, split='train', shuffle=True)
val_dataloader = MRGDataLoader(cfgs, token, split='val', shuffle=False)

optimizer = build_optimizer(cfgs, model)
lr_scheduler = build_lr_scheduler(cfgs, optimizer, len(train_dataloader))

work = MRGPipeline(model, cfgs, 
                criterion=compute_lm_loss, 
                metric_caculator=MetricCalculator(cfgs),
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader
                )

work.train()

from hcpcvlr.utils.cfgs_loader import load_yaml
from hcpcvlr.models.mrg import Baseline
from hcpcvlr.modules.tokenizers import MRGTokenizer
from hcpcvlr.api.pipeline import MRGPipeline
from hcpcvlr.utils.metrics import MetricCalculator
from hcpcvlr.data import MRGDataLoader


cfgs = load_yaml("configs/mrg/baseline.yaml")
token = MRGTokenizer(cfgs)
model = Baseline(cfgs, token)

# -------------------
# inference
# -------------------
work = MRGPipeline(model, cfgs, metric_caculator=MetricCalculator(cfgs))
test_dataloader = MRGDataLoader(cfgs, token, split='test', shuffle=False)
work.inference(test_dataloader)
