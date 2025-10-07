# CausalVLR Installation Guide

## Project Introduction

CausalVLR is a vision-language causal reasoning framework that integrates Medical Report Generation (MRG) and Video Question Answering (VQA) tasks.

### Medical Report Generation (MRG)

Based on CMCRL, generates diagnostic reports from medical images.

Supported models:

- Baseline: Basic encoder-decoder architecture
- VLCI: Vision-Language Causal Intervention model
- VLP: Vision-Language Pre-training model

Supported datasets: IU X-Ray, MIMIC-CXR

### Video Question Answering (VQA)

Based on CRA-GQA, understands video content and answers questions.

Supported models:

- CRA: Causal Relation Aware model
- TempCLIP: Temporal Contrastive Learning model

Supported datasets: NExT-QA, STAR

### Framework Features

- Unified API supporting both tasks
- Modular design with independently usable components
- Configuration-driven experiment management
- Command-line tools for simplified operations
- Integrated standard evaluation metrics
- Traininging visualization monitoring support

## System Requirements

**Hardware**:

- GPU: NVIDIA, 16GB+ VRAM
- Memory: 32GB+ RAM
- Storage: 100GB+ available space

**Software**:

- OS: Linux / Windows
- Python: 3.8 - 3.11
- CUDA: 11.8+
- cuDNN: 9.1.0+

## Installation Steps

### Method 1: Conda (Recommended)

Clone repository:

```bash
git clone https://github.com/yourusername/CausalVLR.git
cd CausalVLR
```

Create environment:

```bash
conda env create -f requirements.yml
conda activate causalvlr
```

Verify installation:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import causalvlr; print(f'CausalVLR: {causalvlr.__version__}')"
```

### Method 2: Manual Installation

Create environment:

```bash
conda create -n causalvlr python=3.11
conda activate causalvlr
```

Install PyTorch:

```bash
# CUDA 11.8
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

Install dependencies:

```bash
pip install transformers==4.47.1 tokenizers==0.21.0 sentencepiece==0.2.0
pip install scipy pandas h5py matplotlib tqdm pyyaml pillow opencv-python
pip install einops tabulate dominate visdom
```

Install CausalVLR:

```bash
pip install -e .
```

## Data Preparation

### MRG Datasets

#### IU X-Ray

Download data:

```bash
wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz
tar -xzf NLMCXR_png.tgz -C data/iu_xray/images
wget https://github.com/cuhksz-nlp/R2Gen/blob/main/data/iu_xray/annotation.json
```

Directory structure:

```
data/iu_xray/
├── images/
│   ├── CXR1_1_IM-0001-3001.png
│   ├── CXR1_1_IM-0001-4001.png
│   └── ...
└── annotation.json
```

#### MIMIC-CXR

Access permission required at PhysioNet: https://physionet.org/content/mimic-cxr/2.0.0/

Directory structure:

```
data/mimic_cxr/
├── images/
│   ├── p10/
│   ├── p11/
│   └── ...
└── annotation.json
```

### VQA Datasets

#### NExT-QA

Download data: https://doc-doc.github.io/docs/nextqa.html

Directory structure:

```
data/nextgqa/
├── video_feature/
│   └── CLIP_L/
│       ├── video_001.pt
│       ├── video_002.pt
│       └── ...
├── causal_feature/
│   ├── video_001_causal.pt
│   └── ...
├── train.csv
├── val.csv
└── test.csv
```

CSV file format:

```
video_id,question,answer,a0,a1,a2,a3,a4
video_001,"What happened before?","option_a","option_a","option_b","option_c","option_d","option_e"
```

#### STAR

Download data: https://bobbwu.com/STAR/

Directory structure:

```
data/star/
├── video_feature/
│   └── CLIP_L/
│       ├── clip_001.pt
│       ├── clip_002.pt
│       └── ...
├── causal_feature/
│   ├── clip_001_causal.pt
│   └── ...
├── train.csv
├── val.csv
└── test.csv
```

CSV file format:

```
clip_id,question,answer,a0,a1,a2,a3
clip_001,"What is the person doing?","walking","walking","sitting","running","standing"
```

## Configuration Files

### MRG Configuration (JSON)

`configs/MRG/my_config.json`:

```json
{
  "data": {
    "dataset_name": "iu_xray",
    "image_dir": "data/iu_xray/images",
    "ann_path": "data/iu_xray/annotation.json",
    "tokenizer": "ori",
    "max_seq_length": 100,
    "threshold": 10,
    "batch_size": 16
  },
  "model": {
    "model": "vlci",
    "embed_dim": 512,
    "num_heads": 8
  },
  "train": {
    "task": "finetune",
    "epochs": 50,
    "save_period": 1,
    "cuda": "0",
    "result_dir": "results/mrg_vlci"
  }
}
```

### VQA Configuration (YAML)

`configs/VQA/my_config.yml`:

```yaml
dataset:
  name: nextgqa
  csv_path: data/nextgqa
  features_path: data/nextgqa/video_feature/CLIP_L
  batch_size: 32

model:
  name: CRA
  baseline: refine
  lan: RoBERTa
  lan_weight_path: "pretrained/roberta-base"

optim:
  pipeline: CRA
  epochs: 20
  lr: 0.0001

misc:
  cuda: "0"
  result_dir: results/vqa_cra
```

## Verify Installation

Testing import:

```bash
python -c "
from causalvlr import build_tokenizer, inference
from causalvlr.api.pipeline.MRG import MRGPipeline
from causalvlr.api.pipeline.VQA import CRAPipeline
print('Import Successfully')
"
```

Check help:

```bash
python -m causalvlr.api.run --help
```

## FAQ

**CUDA out of memory**
Reduce batch_size:

```json
"train": {
    "batch_size": 8
}
```

**Can not find pretrained model**
Set cache path or use local path:

```bash
export HF_HOME=/path/to/huggingface/cache
```

**Visdom connection failed**
Start server:

```bash
python -m visdom.server -port 8097
```

**Import error**
Reinstall:

```bash
pip install -e .
```
