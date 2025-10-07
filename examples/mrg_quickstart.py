"""
MRG Quick Start Example

This example demonstrates how to quickly use CausalVLR for medical report generation.
"""

import json
import torch
from causalvlr.api.pipeline.MRG import MRGPipeline


def main():
    # Prepare Configuration
    config = {
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
            "model": "vlci",           # Use VLCI model
            "embed_dim": 512,
            "v_causal": "y",           # Enable visual causal intervention
            "l_causal": "y",           # Enable language causal intervention
            "num_heads": 8,
            "en_num_layers": 3,
            "de_num_layers": 3,
            "dropout": 0.1,
            "logit_layers": 1,
            "bos_idx": 0,
            "eos_idx": 0,
            "pad_idx": 0,
            "use_bn": 0,
            "drop_prob_lm": 0.5
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
            "result_dir": "results/mrg_quickstart",
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

    # Create Pipeline
    print("\nCreating MRG Pipeline...")
    pipeline = MRGPipeline(config)
    print(f"Model parameters: {sum(p.numel() for p in pipeline.model.parameters()):,}")

    # Training Model
    print("\nStarting training...")
    print(f"Epochs: {config['train']['epochs']}")
    print(f"Learning rate: {config['train']['lr']}")
    print(f"Batch size: {config['data']['batch_size']}")
    print(f"Save directory: {config['train']['result_dir']}")
    
    # Run training
    pipeline.train()
    
    # Evaluate Model
    print("\nEvaluating on test set...")
    results = pipeline.inference()
    
    # Print results
    print("Testing Results")
    for metric, value in results['metrics'].items():
        print(f"{metric:12s}: {value:.4f}")
    
    # Save results
    output_file = f"{config['train']['result_dir']}/test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Show Examples
    print("\nGenerated Examples")
    for i in range(min(3, len(results['predictions']))):
        print(f"\nExample {i+1}:")
        print(f"Ground truth: {results['ground_truth'][i]}")
        print(f"Generated: {results['predictions'][i]}")


if __name__ == "__main__":
    # Check CUDA
    if not torch.cuda.is_available():
        print("Warning: CUDA not detected, will use CPU for training (slower)")
    else:
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    
    main()
