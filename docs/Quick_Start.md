# CausalVLR Quick Start

## Quick Experience

### Using Unified API

```python
from causalvlr import build_tokenizer, inference
import json

# Load configuration
with open('configs/MRG/vlp.json', 'r') as f:
    config = json.load(f)

# Build tokenizer
tokenizer = build_tokenizer(config, task='MRG')

# Run inference
results = inference(config, checkpoint_path='checkpoints/best_model.pth', task='MRG')
```

### Using Command Line

```bash
# MRG training
python -m causalvlr.api.run --config configs/MRG/vlp.json --mode train

# VQA inference
python -m causalvlr.api.run --config configs/VQA/CRA/CRA_NextGQA.yml --mode inference
```

## MRG Quick Start

### Prepare Configuration

Create `my_mrg_config.json`:

```json
{
  "data": {
    "dataset_name": "iu_xray",
    "image_dir": "data/iu_xray/images",
    "ann_path": "data/iu_xray/annotation.json",
    "tokenizer": "ori",
    "max_seq_length": 100,
    "threshold": 10,
    "num_workers": 2,
    "batch_size": 16
  },
  "model": {
    "model": "vlci",
    "embed_dim": 512,
    "v_causal": "y",
    "l_causal": "y",
    "num_heads": 8,
    "en_num_layers": 3,
    "de_num_layers": 3,
    "dropout": 0.1
  },
  "train": {
    "task": "finetune",
    "epochs": 50,
    "lr": 5e-5,
    "weight_decay": 5e-5,
    "lr_scheduler": "StepLR",
    "step_size": 10,
    "gamma": 0.8,
    "save_period": 1,
    "monitor_mode": "max",
    "monitor_metric": "BLEU_4",
    "early_stop": 20,
    "cuda": "0",
    "result_dir": "results/mrg_vlci",
    "resume": "",
    "load_model_path": ""
  },
  "sample": {
    "sample_method": "beam_search",
    "beam_size": 3,
    "n_best": 1
  },
  "loss": {
    "loss_fn": "lm"
  }
}
```

### Method 1: Command Line

```bash
# Traininging
python -m causalvlr.api.run \
    --config my_mrg_config.json \
    --mode train \
    --cuda 0

# Inference
python -m causalvlr.api.run \
    --config my_mrg_config.json \
    --mode inference
```

### Method 2: Python API

```python
from causalvlr.api.pipeline.MRG import MRGPipeline
import json

with open('my_mrg_config.json', 'r') as f:
    config = json.load(f)

pipeline = MRGPipeline(config)

# Training
pipeline.train()

# Inference
results = pipeline.inference()
```

### Method 3: Unified Entry

```python
from causalvlr import inference
import json

with open('my_mrg_config.json', 'r') as f:
    config = json.load(f)

results = inference(
    config=config,
    checkpoint_path='results/mrg_vlci/best_model.pth',
    task='MRG'
)

print(f"Generated reports: {len(results['predictions'])}")
print(f"Metrics: {results['metrics']}")
```

### Check Results

```python
import pandas as pd

log_df = pd.read_csv('results/mrg_vlci/log.csv')
print(log_df.tail())

print(f"Best BLEU-4: {log_df['val_BLEU_4'].max()}")
print(f"Best CIDEr: {log_df['val_CIDEr'].max()}")
```

## VQA Quick Start

### Prepare Configuration

Create `my_vqa_config.yml`:

```yaml
dataset:
  name: nextgqa
  csv_path: data/nextgqa
  features_path: data/nextgqa/video_feature/CLIP_L
  causal_feature_path: data/nextgqa/causal_feature
  batch_size: 32
  num_thread_reader: 4
  qmax_words: 30
  amax_words: 38
  max_feats: 32
  mc: 5
  feat_type: CLIPL

model:
  name: CRA
  baseline: refine
  lan: RoBERTa
  lan_weight_path: "pretrained/roberta-base"
  feature_dim: 768
  word_dim: 768
  num_layers: 2
  num_heads: 8
  d_model: 768
  dropout: 0.3

optim:
  pipeline: CRA
  epochs: 20
  lr: 0.0001
  warmup_proportion: 0.1
  batch_size: 32
  save_period: 1
  print_iter: 100

stat:
  monitor:
    mode: max
    metric: Acc
  early_stop: 10

misc:
  cuda: "0"
  seed: 42
  result_dir: results/vqa_cra
```

### Method 1: Command Line

```bash
# Traininging
python -m causalvlr.api.run \
    --config my_vqa_config.yml \
    --mode train \
    --cuda 0

# Inference
python -m causalvlr.api.run \
    --config my_vqa_config.yml \
    --mode inference
```

### Method 2: Python API

```python
from causalvlr.api.pipeline.VQA import CRAPipeline
import yaml

with open('my_vqa_config.yml', 'r') as f:
    config = yaml.safe_load(f)

pipeline = CRAPipeline(config)

# Training
pipeline.train()

# Inference
results = pipeline.inference()
```

### Method 3: Unified Entry

```python
from causalvlr import inference
import yaml

with open('my_vqa_config.yml', 'r') as f:
    config = yaml.safe_load(f)

results = inference(config, task='VQA')
print(f"Accuracy: {results['accuracy']}")
```

## Workflow Examples

### Training from Scratch

```bash
python scripts/prepare_data.py --dataset iu_xray
python -m causalvlr.api.run --config configs/MRG/vlci.json --mode train
python -m causalvlr.api.run --config configs/MRG/vlci.json --mode inference
python scripts/analyze_results.py --result_dir results/mrg_vlci
```

### Using Pretrained Model

```python
from causalvlr.api.pipeline.MRG import MRGPipeline
import json

with open('configs/MRG/vlp.json', 'r') as f:
    config = json.load(f)

config['train']['load_model_path'] = 'pretrained/vlp_pretrained.pth'

pipeline = MRGPipeline(config)
pipeline.train()
```

### Hyperparameter Search

```python
import itertools
from causalvlr.api.pipeline.MRG import MRGPipeline

lr_list = [1e-4, 5e-5, 1e-5]
batch_size_list = [8, 16, 32]

for lr, bs in itertools.product(lr_list, batch_size_list):
    config['train']['lr'] = lr
    config['data']['batch_size'] = bs
    config['train']['result_dir'] = f'results/search_lr{lr}_bs{bs}'

    pipeline = MRGPipeline(config)
    metrics = pipeline.train()

    print(f"LR={lr}, BS={bs}, BLEU-4={metrics['best_bleu4']}")
```

## Monitor Traininging

### Using Visdom

```bash
python -m visdom.server -port 8097

# Add to configuration:
{
    "train": {
        "monitor": true,
        "monitor_port": 8097
    }
}

# Access http://localhost:8097
```

### Check Logs

```python
import pandas as pd
import matplotlib.pyplot as plt

log = pd.read_csv('results/mrg_vlci/log.csv')

plt.figure(figsize=(10, 6))
plt.plot(log['epoch'], log['train_loss'], label='Training Loss')
plt.plot(log['epoch'], log['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve.png')

plt.figure(figsize=(10, 6))
plt.plot(log['epoch'], log['val_BLEU_4'], label='BLEU-4')
plt.plot(log['epoch'], log['val_CIDEr'], label='CIDEr')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.savefig('metrics_curve.png')
```

## Single Sample Testinging

### MRG Inference

```python
from causalvlr.api.pipeline.MRG import MRGPipeline
from PIL import Image
import torch

pipeline = MRGPipeline(config)
pipeline.model.load_state_dict(torch.load('best_model.pth'))
pipeline.model.eval()

image = Image.open('test_image.png')
image_tensor = pipeline.transform(image).unsqueeze(0).cuda()

with torch.no_grad():
    report = pipeline.model.generate(image_tensor, pipeline.tokenizer)

print(f"Generated report: {report}")
```

### VQA Inference

```python
from causalvlr.api.pipeline.VQA import CRAPipeline
import torch

pipeline = CRAPipeline(config)
pipeline.load_checkpoint('best_model.pth')

video_features = torch.load('video_features.pt')
question = "What happens after the person opens the door?"

answer = pipeline.predict_single(video_features, question)
print(f"Predicted answer: {answer}")
```

## Practical Tips

### Quick Debugging

```python
config['data']['batch_size'] = 2
config['train']['epochs'] = 2
config['data']['num_workers'] = 0

pipeline = MRGPipeline(config)
pipeline.train()
```

### Resume Traininging

```python
config['train']['resume'] = 'results/mrg_vlci/checkpoint_epoch_10.pth'

pipeline = MRGPipeline(config)
pipeline.train()
```

### Mixed Precision Traininging

```python
config['train']['use_amp'] = True
config['train']['amp_opt_level'] = 'O1'
```

### Multi-GPU Traininging

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m causalvlr.api.run --config config.json --mode train
```

## Common Code Snippets

### Check Model Parameters

```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {count_parameters(pipeline.model):,}")
```

### Fix Random Seed

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
```

### Save Generated Results

```python
import json
import pandas as pd

with open('generated_reports.json', 'w') as f:
    json.dump(results, f, indent=2)

df = pd.DataFrame(results['reports'])
df.to_csv('generated_reports.csv', index=False)
```
