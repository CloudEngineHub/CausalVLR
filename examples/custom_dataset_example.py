"""
Custom Dataset Example

Demonstrates how to create and use custom datasets.
"""

import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from causalvlr.data.MRG import BaseDataset, Tokenizer
from causalvlr.api.pipeline.MRG import MRGPipeline


class CustomMRGDataset(BaseDataset):
    """Custom MRG Dataset"""
    
    def __init__(self, args, tokenizer, split='train'):
        """
        Args:
            args: Configuration dictionary
            tokenizer: Tokenizer object
            split: 'train', 'val', or 'test'
        """
        super().__init__(args, tokenizer, split)
        
        # Load custom annotation file
        ann_path = args.get('custom_ann_path', args['ann_path'])
        with open(ann_path, 'r') as f:
            self.annotations = json.load(f)
        
        # Filter data by split
        self.examples = self.annotations[split]
        
        print(f"Loaded {split} set: {len(self.examples)} samples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        example = self.examples[idx]
        
        # 1. Load image
        image_path = example['image_path']
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.split == 'train':
            image = self.transform_train(image)
        else:
            image = self.transform_test(image)
        
        # 2. Process report text
        report = example['report']
        
        # Clean text
        report_clean = self.tokenizer.clean_report(report)
        
        # Encode to token IDs
        report_ids = self.tokenizer.encode(report_clean)
        
        # Truncate or pad to fixed length
        max_len = self.args.get('max_seq_length', 100)
        if len(report_ids) > max_len:
            report_ids = report_ids[:max_len]
        else:
            report_ids = report_ids + [self.tokenizer.pad_idx] * (max_len - len(report_ids))
        
        # 3. Create mask
        mask = [1 if tid != self.tokenizer.pad_idx else 0 for tid in report_ids]
        
        # 4. Additional information (optional)
        metadata = {
            'image_id': example.get('image_id', idx),
            'patient_id': example.get('patient_id', ''),
            'study_date': example.get('study_date', '')
        }
        
        return {
            'images': image,                    # [C, H, W]
            'reports': torch.LongTensor(report_ids),  # [max_len]
            'masks': torch.FloatTensor(mask),   # [max_len]
            'metadata': metadata
        }


class MultiImageMRGDataset(BaseDataset):
    """MRG dataset supporting multiple images"""
    
    def __init__(self, args, tokenizer, split='train'):
        super().__init__(args, tokenizer, split)
        
        with open(args['ann_path'], 'r') as f:
            self.annotations = json.load(f)[split]
        
        self.max_images = args.get('max_images', 2)  # Maximum number of images to use
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        example = self.annotations[idx]
        
        # Load multiple images
        image_paths = example['image_paths']  # List
        images = []
        
        for img_path in image_paths[:self.max_images]:
            image = Image.open(img_path).convert('RGB')
            if self.split == 'train':
                image = self.transform_train(image)
            else:
                image = self.transform_test(image)
            images.append(image)
        
        # Pad with zeros if insufficient images
        while len(images) < self.max_images:
            images.append(torch.zeros_like(images[0]))
        
        # Stack images
        images_tensor = torch.stack(images)  # [max_images, C, H, W]
        
        # Process report
        report = example['report']
        report_clean = self.tokenizer.clean_report(report)
        report_ids = self.tokenizer.encode(report_clean)
        
        max_len = self.args.get('max_seq_length', 100)
        if len(report_ids) > max_len:
            report_ids = report_ids[:max_len]
        else:
            report_ids = report_ids + [self.tokenizer.pad_idx] * (max_len - len(report_ids))
        
        mask = [1 if tid != self.tokenizer.pad_idx else 0 for tid in report_ids]
        
        return {
            'images': images_tensor,
            'reports': torch.LongTensor(report_ids),
            'masks': torch.FloatTensor(mask),
            'num_images': len(image_paths)  # Actual number of images
        }


def create_custom_annotation():
    """Create custom annotation file example"""
    
    # Example annotation structure
    annotations = {
        "train": [
            {
                "image_id": "img_001",
                "image_path": "data/custom/images/img_001.png",
                "report": "Findings: Clear lungs. Normal heart size.",
                "patient_id": "P001",
                "study_date": "2025-10-01"
            },
            {
                "image_id": "img_002",
                "image_path": "data/custom/images/img_002.png",
                "report": "Findings: Small pleural effusion on the right.",
                "patient_id": "P002",
                "study_date": "2025-10-01"
            }
        ],
        "val": [
            {
                "image_id": "img_003",
                "image_path": "data/custom/images/img_003.png",
                "report": "Findings: Mild cardiomegaly.",
                "patient_id": "P003",
                "study_date": "2025-10-01"
            }
        ],
        "test": [
            {
                "image_id": "img_004",
                "image_path": "data/custom/images/img_004.png",
                "report": "Findings: No acute findings.",
                "patient_id": "P004",
                "study_date": "2025-10-01"
            }
        ]
    }
    
    # Save annotation file
    output_path = "data/custom/custom_annotations.json"
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Custom annotation file created: {output_path}")
    return output_path


def use_custom_dataset():
    """Training using custom dataset"""
    
    # Configuration
    config = {
        "data": {
            "dataset_name": "custom",
            "image_dir": "data/custom/images",
            "ann_path": "data/custom/annotations.json",
            "custom_ann_path": "data/custom/custom_annotations.json",
            "tokenizer": "ori",
            "max_seq_length": 100,
            "threshold": 10,
            "batch_size": 8
        },
        "model": {
            "model": "baseline",
            "embed_dim": 512,
            "num_heads": 8,
            "en_num_layers": 3,
            "de_num_layers": 3
        },
        "train": {
            "task": "finetune",
            "epochs": 10,
            "lr": 1e-4,
            "result_dir": "results/custom_dataset"
        }
    }
    
    # Note: Need to modify MRGPipeline to support custom datasets
    # This shows how to manually create data loaders
    
    from causalvlr.utils.MRG import tokenizers_fn
    from torch.utils.data import DataLoader
    
    # Create tokenizer
    tokenizer = tokenizers_fn['ori'](config['data'])
    
    # Create custom datasets
    train_dataset = CustomMRGDataset(config['data'], tokenizer, split='train')
    val_dataset = CustomMRGDataset(config['data'], tokenizer, split='val')
    test_dataset = CustomMRGDataset(config['data'], tokenizer, split='test')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    print(f"Data loaders created successfully")
    print(f"  Training set: {len(train_dataset)} samples")
    print(f"  Val set: {len(val_dataset)} samples")
    print(f"  Testing set: {len(test_dataset)} samples")
    
    # Testing data loading
    print("\nTestinging data loading...")
    for batch in train_loader:
        print(f"  Batch images shape: {batch['images'].shape}")
        print(f"  Batch reports shape: {batch['reports'].shape}")
        print(f"  Batch masks shape: {batch['masks'].shape}")
        break
    
    return train_loader, val_loader, test_dataset


def main():

    # Create custom annotation file
    print("\nCreating custom annotation file...")
    ann_path = create_custom_annotation()
    
    # Use custom dataset
    print("\nUsing custom dataset...")
    try:
        train_loader, val_loader, test_dataset = use_custom_dataset()
        print("\nCustom dataset test successful!")
    except Exception as e:
        print("Please ensure data files exist and paths are correct")
    
    print("Example complete")


if __name__ == "__main__":
    main()
