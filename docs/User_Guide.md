# CausalVLR User Guide

This guide provides detailed introduction to CausalVLR framework modules and their usage.

---

## Table of Contents

1. [Core Components](#core-components)
2. [Medical Report Generation (MRG)](#medical-report-generation-mrg)
3. [Video Question Answering (VQA)](#video-question-answering-vqa)
4. [Data Processing](#data-processing)
5. [Model Traininging](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Model Deployment](#model-deployment)

---

## Core Components

### Unified Tokenizer

CausalVLR provides a unified tokenizer building interface that automatically adapts to different tasks:

```python
from causalvlr import build_tokenizer

# Method 1: Auto-detect task type
tokenizer = build_tokenizer(config)

# Method 2: Explicitly specify task type
tokenizer_mrg = build_tokenizer(config, task='MRG')
tokenizer_vqa = build_tokenizer(config, task='VQA')

# MRG tokenizer usage
token_ids = tokenizer_mrg.encode("Findings: No acute cardiopulmonary process.")
text = tokenizer_mrg.decode(token_ids)

# VQA tokenizer usage (based on BERT/RoBERTa)
encoded = tokenizer_vqa("What is the person doing?", padding=True, truncation=True)
```

### Unified Inference Interface

```python
from causalvlr import inference
import json
import yaml

# MRG inference
with open('configs/MRG/vlci.json', 'r') as f:
    mrg_config = json.load(f)

mrg_results = inference(
    config=mrg_config,
    checkpoint_path='checkpoints/mrg_best.pth',
    task='MRG'
)

# VQA inference
with open('configs/VQA/CRA/CRA_NextGQA.yml', 'r') as f:
    vqa_config = yaml.safe_load(f)

vqa_results = inference(
    config=vqa_config,
    checkpoint_path='checkpoints/vqa_best.pth',
    task='VQA'
)
```

### Command Line Tools

```bash
# Check help
python -m causalvlr.api.run --help

# Traininging
python -m causalvlr.api.run \
    --config configs/MRG/vlci.json \
    --mode train \
    --cuda 0 \
    --seed 42

# Inference
python -m causalvlr.api.run \
    --config configs/VQA/CRA/CRA_NextGQA.yml \
    --mode inference \
    --cuda 0
```

---

## Medical Report Generation (MRG)

### Supported Models

#### 1. Baseline Model

Basic encoder-decoder architecture:

```json
{
  "model": {
    "model": "baseline",
    "embed_dim": 512,
    "num_heads": 8,
    "en_num_layers": 3,
    "de_num_layers": 3
  }
}
```

#### 2. VLCI Model

Integrated vision-language causal intervention:

```json
{
  "model": {
    "model": "vlci",
    "embed_dim": 512,
    "v_causal": "y", // Visual causal intervention
    "l_causal": "y", // Language causal intervention
    "num_heads": 8,
    "en_num_layers": 3,
    "de_num_layers": 3
  }
}
```

#### 3. VLP Model

Vision-language pre-training model:

```json
{
  "model": {
    "model": "vlp",
    "embed_dim": 512,
    "v_mask_ratio": 0.85, // Visual masking ratio
    "num_heads": 8,
    "en_num_layers": 3,
    "de_num_layers": 3
  }
}
```

### Complete Traininging Workflow

#### Stage 1: Pre-training (Optional)

```python
from causalvlr.api.pipeline.MRG import MRGPipeline
import json

# Pre-training configuration
pretrain_config = {
    "data": {
        "dataset_name": "mimic_cxr",
        "image_dir": "data/mimic_cxr/images",
        "ann_path": "data/mimic_cxr/annotation.json",
        "batch_size": 64
    },
    "model": {
        "model": "vlp",
        "v_mask_ratio": 0.85
    },
    "train": {
        "task": "pretrain",  # Pre-training task
        "epochs": 100,
        "lr": 1e-4,
        "result_dir": "results/pretrain_vlp"
    }
}

# Run pre-training
pipeline = MRGPipeline(pretrain_config)
pipeline.train()
```

#### Stage 2: Fine-tuning

```python
# Fine-tuning configuration
finetune_config = {
    "data": {
        "dataset_name": "iu_xray",
        "image_dir": "data/iu_xray/images",
        "ann_path": "data/iu_xray/annotation.json",
        "batch_size": 16
    },
    "model": {
        "model": "vlp"
    },
    "train": {
        "task": "finetune",  # Fine-tuning task
        "epochs": 50,
        "lr": 5e-5,
        "load_model_path": "results/pretrain_vlp/best_model.pth",  # Load pre-trained weights
        "result_dir": "results/finetune_vlp"
    }
}

# Run fine-tuning
pipeline = MRGPipeline(finetune_config)
pipeline.train()
```

### Custom Dataset

```python
from causalvlr.data.MRG import BaseDataset
from PIL import Image
import json

class CustomMRGDataset(BaseDataset):
    def __init__(self, args, tokenizer, split='train'):
        super().__init__(args, tokenizer, split)

        # Load custom annotations
        with open(args['custom_ann_path'], 'r') as f:
            self.annotations = json.load(f)

    def __getitem__(self, idx):
        ann = self.annotations[idx]

        # Load image
        image_path = ann['image_path']
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)

        # Encode report
        report = ann['report']
        report_ids = self.tokenizer.encode(report)

        return {
            'images': image_tensor,
            'reports': report_ids,
            'image_id': ann['image_id']
        }

# Use custom dataset in configuration
config['data']['dataset_class'] = 'CustomMRGDataset'
config['data']['custom_ann_path'] = 'data/custom/annotations.json'
```

### Generation Strategies

#### Beam Search

```json
{
  "sample": {
    "sample_method": "beam_search",
    "beam_size": 3,
    "n_best": 1,
    "length_penalty": 0.6,
    "early_stopping": true
  }
}
```

#### Greedy Decoding

```json
{
  "sample": {
    "sample_method": "greedy",
    "max_length": 100
  }
}
```

#### Sampling

```json
{
  "sample": {
    "sample_method": "sampling",
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 0.9
  }
}
```

### Evaluation Metrics

```python
from causalvlr.metrics.mrg_metric.metrics import compute_scores

# Prepare data
ground_truth = {
    'img1': ['Normal chest X-ray.', 'No acute findings.'],
    'img2': ['Cardiomegaly present.']
}

predictions = {
    'img1': ['Normal chest radiograph.'],
    'img2': ['Enlarged cardiac silhouette.']
}

# Compute metrics
scores = compute_scores(ground_truth, predictions)
print(f"BLEU-1: {scores['BLEU_1']:.4f}")
print(f"BLEU-4: {scores['BLEU_4']:.4f}")
print(f"METEOR: {scores['METEOR']:.4f}")
print(f"ROUGE-L: {scores['ROUGE_L']:.4f}")
print(f"CIDEr: {scores['CIDEr']:.4f}")
```

---

## Video Question Answering (VQA)

### Supported Models

#### 1. CRA Model

Causal relation aware video question answering:

```yaml
model:
  name: CRA
  baseline: refine # Baseline type: refine/finetune
  lan: RoBERTa # Language model: RoBERTa/BERT
  lan_weight_path: "pretrained/roberta-base"
  feature_dim: 768
  word_dim: 768
  num_layers: 2
  num_heads: 8
  d_model: 768
  d_ff: 768
  dropout: 0.3
  n_negs: 1 # Number of negative samples
```

#### 2. TempCLIP Model

Based on temporal contrastive learning:

```yaml
model:
  name: TempCLIP
  baseline: refine
  lan: RoBERTa
  lan_weight_path: "pretrained/roberta-base"
  feature_dim: 768
  word_dim: 768
  temporal_layers: 3 # Number of temporal layers
  contrast_temp: 0.07 # Contrastive learning temperature
```

### Complete Traininging Workflow

```python
from causalvlr.api.pipeline.VQA import CRAPipeline
import yaml

# Load configuration
with open('configs/VQA/CRA/CRA_NextGQA.yml', 'r') as f:
    config = yaml.safe_load(f)

# Create pipeline
pipeline = CRAPipeline(config)

# Training
train_results = pipeline.train()
print(f"Best validation accuracy: {train_results['best_val_acc']:.4f}")

# Testing
test_results = pipeline.inference()
print(f"Test accuracy: {test_results['test_acc']:.4f}")
```

### Data Preprocessing

#### Video Feature Extraction

```python
import torch
import clip
from PIL import Image
import cv2

def extract_clip_features(video_path, num_frames=32):
    """Extract video features using CLIP"""

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)

    # Read video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Uniform sampling frames
    frame_indices = torch.linspace(0, total_frames - 1, num_frames).long()

    features = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx.item())
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        # Extract features
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model.encode_image(image_input)
        features.append(feature.cpu())

    cap.release()

    # Concatenate features
    video_features = torch.cat(features, dim=0)  # [num_frames, feature_dim]
    return video_features

# Batch processing
video_list = ['video1.mp4', 'video2.mp4', 'video3.mp4']
for video_path in video_list:
    features = extract_clip_features(video_path)
    save_path = video_path.replace('.mp4', '_clip_features.pt')
    torch.save(features, save_path)
```

#### Causal Feature Extraction

```python
from causalvlr.data.VQA import prepare_causal_features

def extract_causal_features(video_features, question):
    """Extract causal-related video features"""

    # Implement causal feature extraction logic
    # Simplified example
    causal_features = prepare_causal_features(
        video_features=video_features,
        question_text=question,
        method='attention'  # or 'graph', 'intervention'
    )

    return causal_features
```

### Multiple Choice vs Open-ended QA

#### Multiple Choice Configuration (NExT-QA, STAR)

```yaml
dataset:
  mc: 5 # 5 candidate answers
  a2id: null # No answer vocabulary needed
```

#### Open-ended QA Configuration

```yaml
dataset:
  mc: 0 # Disable multiple choice
  vocab_path: data/vocab.json # Answer vocabulary
  amax_words: 38 # Maximum answer length
```

### Evaluation Metrics

```python
from causalvlr.utils.VQA import Metric

# Create evaluator
metric = Metric(config)

# Add predictions
for batch in test_loader:
    predictions = model(batch)
    metric.update(predictions, batch['answers'])

# Get final results
results = metric.compute()
print(f"Accuracy: {results['Acc']:.4f}")
print(f"Top-5 Accuracy: {results['Acc@5']:.4f}")
```

---

## Data Processing

### Data Loaders

#### MRG DataLoader

```python
from causalvlr.data.MRG import R2DataLoader

# Create data loaders
train_loader = R2DataLoader(
    args=config,
    tokenizer=tokenizer,
    split='train',
    shuffle=True
)

val_loader = R2DataLoader(
    args=config,
    tokenizer=tokenizer,
    split='val',
    shuffle=False
)

# Iterate data
for batch in train_loader:
    images = batch['images']      # [B, C, H, W]
    reports = batch['reports']    # [B, max_len]
    masks = batch['masks']        # [B, max_len]

    # Trainingcode...
```

#### VQA DataLoader

```python
from causalvlr.data.VQA import build_dataloaders

# Build all data loaders
train_loader, val_loader, test_loader = build_dataloaders(
    config,
    a2id=answer_vocab,
    tokenizer=tokenizer
)

# Iterate data
for batch in train_loader:
    video_features = batch['video']      # [B, T, D]
    questions = batch['question']        # [B, L]
    answers = batch['answer']            # [B, num_choices] or [B]

    # Trainingcode...
```

### Data Augmentation

#### Image Augmentation (MRG)

```python
from torchvision import transforms

# Training augmentation
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Testing augmentation
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
```

#### Video Augmentation (VQA)

```python
from causalvlr.data.VQA import video_sampling

# Temporal sampling
sampled_features = video_sampling(
    video_features,      # [T, D]
    num_frames=32,       # Target number of frames
    sampling='uniform'   # 'uniform', 'random', 'dense'
)

# Feature augmentation
augmented_features = feature_augmentation(
    sampled_features,
    dropout=0.1,
    noise_std=0.01
)
```

---

## Model Traininging

### Using Trainingers

#### MRG Traininger

```python
from causalvlr.trainer.mrg import FTraininger, PTraininger

# Finetune Traininger
trainer = FTraininger(
    model=model,
    criterion=criterion,
    metric_ftns=compute_scores,
    optimizer=optimizer,
    args=config,
    lr_scheduler=lr_scheduler,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    test_dataloader=test_loader
)

# Start traininging
trainer.train()

# Pretrain Trainer (for pre-training tasks)
pretrainer = PTraininger(
    model=model,
    criterion=criterion,
    metric_ftns=compute_scores,
    optimizer=optimizer,
    args=config,
    lr_scheduler=lr_scheduler,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    test_dataloader=test_loader
)

pretrainer.train()
```

### Optimizer Configuration

```python
from causalvlr.utils.MRG import build_optimizer, build_lr_scheduler

# Build optimizer
optimizer = build_optimizer(config, model)

# Supported optimizers
config['train']['optimizer'] = 'Adam'      # Adam
config['train']['optimizer'] = 'AdamW'     # AdamW
config['train']['optimizer'] = 'SGD'       # SGD

# Optimizer parameters
config['train']['lr'] = 5e-5
config['train']['weight_decay'] = 5e-5
config['train']['betas'] = [0.9, 0.999]    # for Adam/AdamW

# Build learning rate scheduler
lr_scheduler = build_lr_scheduler(config, optimizer, len(train_loader))

# Supported schedulers
config['train']['lr_scheduler'] = 'StepLR'           # Step decay
config['train']['lr_scheduler'] = 'CosineAnnealing'  # Cosine annealing
config['train']['lr_scheduler'] = 'ReduceLROnPlateau'# Adaptive decay
```

### Loss Functions

#### MRG Loss

```python
from causalvlr.utils.MRG import compute_lm_loss, compute_recon_loss

# Language model loss (for report generation)
lm_loss = compute_lm_loss(
    predictions,  # [B, L, vocab_size]
    targets,      # [B, L]
    masks         # [B, L]
)

# Reconstruction loss (for pre-training)
recon_loss = compute_recon_loss(
    reconstructed,  # [B, C, H, W]
    original,       # [B, C, H, W]
    mask           # [B, H, W]
)
```

#### VQA Loss

```python
from causalvlr.utils.VQA import BuildLossFunc

# Build loss function
criterion = BuildLossFunc(config)

# Multiple choice cross-entropy loss
loss = criterion(
    logits,   # [B, num_choices]
    labels    # [B]
)
```

### Traininging Monitoring

```python
from causalvlr.utils.MRG import Monitor

# Create monitor
monitor = Monitor(
    server='localhost',
    port=8097,
    env='mrg_experiment'
)

# Log scalars
monitor.plot('loss', train_loss, epoch, 'train')
monitor.plot('loss', val_loss, epoch, 'val')

# Log images
monitor.image(sample_image, caption='Generated Report')

# Log text
monitor.text(generated_report, 'Sample Generation')
```

### Early Stopping

```json
{
  "train": {
    "monitor_mode": "max", // 'max' or 'min'
    "monitor_metric": "BLEU_4", // Monitored metric
    "early_stop": 20, // Early stopping rounds
    "save_period": 1 // Save period
  }
}
```

---

## Model Evaluation

### Evaluation Script

```python
from causalvlr.api.pipeline.MRG import MRGPipeline
import json

# Load model
config['train']['load_model_path'] = 'results/best_model.pth'
pipeline = MRGPipeline(config)

# Evaluate on test set
test_results = pipeline.inference()

# Save results
with open('test_results.json', 'w') as f:
    json.dump(test_results, f, indent=2)

# Print metrics
print("=== Testing Results ===")
for metric, value in test_results['metrics'].items():
    print(f"{metric}: {value:.4f}")
```

### Generated Report Analysis

```python
import pandas as pd

# Load generated reports
results = json.load(open('test_results.json'))

# Convert to DataFrame
df = pd.DataFrame({
    'image_id': results['image_ids'],
    'ground_truth': results['ground_truth'],
    'prediction': results['predictions']
})

# Analyze report length
df['gt_length'] = df['ground_truth'].apply(lambda x: len(x.split()))
df['pred_length'] = df['prediction'].apply(lambda x: len(x.split()))

print(f"Average ground truth length: {df['gt_length'].mean():.1f}")
print(f"Average generated length: {df['pred_length'].mean():.1f}")

# Check samples
print("\n=== Sample Generations ===")
for i in range(5):
    print(f"\nSample {i+1}:")
    print(f"Ground truth: {df.iloc[i]['ground_truth']}")
    print(f"Prediction: {df.iloc[i]['prediction']}")
```

### Error Analysis

```python
def analyze_errors(results, threshold=0.3):
    """Analyze low-score samples"""

    errors = []
    for i, (gt, pred, score) in enumerate(zip(
        results['ground_truth'],
        results['predictions'],
        results['scores']
    )):
        if score < threshold:
            errors.append({
                'index': i,
                'ground_truth': gt,
                'prediction': pred,
                'score': score
            })

    # Sort by score
    errors.sort(key=lambda x: x['score'])

    return errors

# Get low-score samples
low_score_samples = analyze_errors(test_results, threshold=0.3)
print(f"Found {len(low_score_samples)} low-score samples")

# Check worst 5
for sample in low_score_samples[:5]:
    print(f"\nScore: {sample['score']:.4f}")
    print(f"Ground truth: {sample['ground_truth']}")
    print(f"Prediction: {sample['prediction']}")
```

---

## Model Deployment

### Model Export

```python
import torch

# Load trained model
pipeline = MRGPipeline(config)
pipeline.model.load_state_dict(torch.load('best_model.pth'))
pipeline.model.eval()

# Export as TorchScript
scripted_model = torch.jit.script(pipeline.model)
scripted_model.save('model_scripted.pt')

# Export as ONNX
dummy_input = torch.randn(1, 3, 224, 224).cuda()
torch.onnx.export(
    pipeline.model,
    dummy_input,
    'model.onnx',
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)
```

### Inference Service

```python
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load model
pipeline = MRGPipeline(config)
pipeline.model.load_state_dict(torch.load('best_model.pth'))
pipeline.model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    # Receive image
    image_data = request.json['image']
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))

    # Preprocess
    image_tensor = pipeline.transform(image).unsqueeze(0).cuda()

    # Generate report
    with torch.no_grad():
        report = pipeline.model.generate(image_tensor, pipeline.tokenizer)

    return jsonify({'report': report})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Batch Inference

```python
def batch_inference(image_dir, model_path, output_path, batch_size=32):
    """Batch inference for all images"""

    from pathlib import Path
    from tqdm import tqdm

    # Load model
    pipeline = MRGPipeline(config)
    pipeline.model.load_state_dict(torch.load(model_path))
    pipeline.model.eval()

    # Get all images
    image_paths = list(Path(image_dir).glob('*.png'))

    results = []
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i+batch_size]

        # Load batch images
        batch_images = []
        for path in batch_paths:
            image = Image.open(path).convert('RGB')
            image_tensor = pipeline.transform(image)
            batch_images.append(image_tensor)

        batch_tensor = torch.stack(batch_images).cuda()

        # Batch generation
        with torch.no_grad():
            reports = pipeline.model.generate_batch(batch_tensor, pipeline.tokenizer)

        # Save results
        for path, report in zip(batch_paths, reports):
            results.append({
                'image_id': path.stem,
                'report': report
            })

    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results

# Run batch inference
results = batch_inference(
    image_dir='data/test_images',
    model_path='best_model.pth',
    output_path='predictions.json'
)
```
