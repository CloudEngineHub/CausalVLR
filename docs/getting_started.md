# Getting started

## Installation

### Requirements

- python (>=3.7)
- pytorch
- numpy
- networkx
- pandas
- scipy
- scikit-learn
- statsmodels
- pydot

(For visualization)
- matplotlib
- graphviz
- pygraphviz

### Install via PyPI
To use CausalVLR, we could install it using pip:

```
 pip install CausalVLR
```

### Install from source
For development version, please kindly refer to our GitHub Repository.

## Running examples
To implement the state-of-the-art causal learning algorithms for various visual-linguistic reasoning tasks, there are various running examples in the [test](../test) directory in our GitHub Repository, such as medical report generantion task (MRG).

We prepare a series of pipelines for various tasks, either by modifying the configuration of the specified model or by building your own methods through the modules we provide. Here is the example of MRG task:

- As for inference:

```
from utils.cfgs_loader import load_yaml
from models.mrg import Baseline
from modules.tokenizers import MRGTokenizer
from api.pipeline import MRGPipeline
from utils.metrics import MetricCalculator
from data import MRGDataLoader


cfgs = load_yaml("configs/mrg/baseline.yaml")
token = MRGTokenizer(cfgs)
model = Baseline(cfgs, token)

# -------------------
# inference
# -------------------
work = MRGPipeline(model, cfgs, metric_caculator=MetricCalculator(cfgs))
test_dataloader = MRGDataLoader(cfgs, token, split='test', shuffle=False)
work.inference(test_dataloader)

```

- As for model training:

```
from models.mrg import VLCI
from modules.losses.nlg import compute_lm_loss
from utils.optimizer import build_lr_scheduler, build_optimizer

# -------------------
# training
# -------------------
cfgs = load_yaml("configs/mrg/vlci.yaml")
token = MRGTokenizer(cfgs)
model = VLCI(cfgs, token)
train_dataloader = MRGDataLoader(cfgs, token, split='train', shuffle=True)
val_dataloader = MRGDataLoader(cfgs, token, split='val', shuffle=False)
test_dataloader = MRGDataLoader(cfgs, token, split='test', shuffle=False)

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
```

For the implemented modules, such as (conditional) independent test methods, we provide unit tests for the convenience of developing your own methods.

## Contributors
Team Leaders: Liang Lin, Guanbin Li, Yang Liu

Coordinators: Weixing Chen

## Citation
Please cite as:

```
@misc{liu2023causalvlr,
      title={CausalVLR: A Toolbox and Benchmark for Visual-Linguistic Causal Reasoning}, 
      author={Yang Liu and Weixing Chen and Guanbin Li and Liang Lin},
      year={2023},
      eprint={2306.17462},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<div align=right>

## [Next](method.md)

</div>