# CausalVLR example code

This directory contains usage examples for the CausalVLR framework.

## Example List

### Quick Start

#### 1. MRG Quick Start - `mrg_quickstart.py`

Basic Medical Report Generation example:

- Create configuration file
- Training using MRGPipeline
- Run inference and evaluation

```bash
python examples/mrg_quickstart.py
```

#### 2. VQA Quick Start - `vqa_quickstart.py`

Basic Video Question Answering example:

- Create configuration file
- Training using CRAPipeline
- Run inference and evaluation

```bash
python examples/vqa_quickstart.py
```

### Unified API Usage

#### 3. Unified API Example - `unified_api_example.py`

Using CausalVLR's unified API interface:

- `build_tokenizer()` Build tokenizer
- `inference()` Unified inference interface
- Command line tool usage
- Batch inference

```bash
python examples/unified_api_example.py
```

Included features:

- MRG Inference
- VQA Inference
- Tokenizer Usage
- Batch Processing

### Data Processing

#### 4. Custom Dataset - `custom_dataset_example.py`

Create and use custom datasets:

- Inherit `BaseDataset` Create custom dataset
- Handle custom annotation format
- Support multi-image input
- Custom data augmentation

```bash
python examples/custom_dataset_example.py
```

Applicable scenarios:

- Use your own medical image data
- Custom annotation format
- Special data preprocessing requirements

## By Task

### Medical Report Generation (MRG)

| Example                       | Difficulty | Description        |
| ----------------------------- | ---------- | ------------------ |
| `mrg_quickstart.py`         | Basic      | MRG Quick Start    |
| `custom_dataset_example.py` | Advanced   | Custom MRG Dataset |
| `unified_api_example.py`    | Basic      | Unified API Usage  |

### Video Question Answering (VQA)

| Example                    | Difficulty | Description       |
| -------------------------- | ---------- | ----------------- |
| `vqa_quickstart.py`      | Basic      | VQA Quick Start   |
| `unified_api_example.py` | Basic      | Unified API Usage |

## Usage Guide

### Before Running

Install CausalVLR:

```bash
pip install -e .
```

Prepare data (if needed):

```bash
# Download sample data
python scripts/download_sample_data.py

# Or use your own data
```

### Run Examples

Run Python script directly:

```bash
python examples/mrg_quickstart.py
```

Or use command line tool:

```bash
python -m causalvlr.api.run --config examples/configs/mrg_example.json --mode train
```

### Modify Configuration

All examples can be adapted by modifying configuration parameters:

```python
config['data']['batch_size'] = 32
config['train']['lr'] = 1e-4
config['train']['epochs'] = 100
```

## Learning Path

### Beginners

1. `mrg_quickstart.py` - Understand basic workflow
2. `unified_api_example.py` - Learn unified API
3. `custom_dataset_example.py` - Use your own data

### Advanced Users

1. `custom_dataset_example.py` - Use your own data
2. Refer to documentation for in-depth framework learning
3. Modify configuration based on needs

## FAQ

**Examples fail to run?**
Check if all dependencies are installed:

```bash
pip install -r requirements.txt
```

**Where to download datasets?**
Refer to `docs/01_Project Introduction and Installation.md` for data preparation section.

**How to use your own data?**
Refer to `custom_dataset_example.py` example.

**How to modify model configuration?**
Modify configuration dictionary or file, refer to `docs/User Guide.md`.
