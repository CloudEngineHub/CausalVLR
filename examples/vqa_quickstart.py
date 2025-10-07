"""
VQA Quick Start Example
Video Question Answering Quick Start Example

This example demonstrates how to quickly use CausalVLR for video question answering.
"""

import yaml
import torch
from causalvlr.api.pipeline.VQA import CRAPipeline


def main():
    # Prepare Configuration
    config = {
        "dataset": {
            "name": "nextgqa",
            "csv_path": "data/nextgqa",
            "features_path": "data/nextgqa/video_feature/CLIP_L",
            "causal_feature_path": "data/nextgqa/causal_feature",
            "vocab_path": None,
            "batch_size": 32,
            "num_thread_reader": 4,
            "qmax_words": 30,
            "amax_words": 38,
            "max_feats": 32,
            "mc": 5,  # Multiple choice
            "a2id": None,
            "feat_type": "CLIPL"
        },
        "model": {
            "name": "CRA",
            "baseline": "refine",
            "lan": "RoBERTa",
            "lan_weight_path": "pretrained/roberta-base",
            "feature_dim": 768,
            "word_dim": 768,
            "num_layers": 2,
            "num_heads": 8,
            "d_model": 768,
            "d_ff": 768,
            "dropout": 0.3,
            "vocab_size": 50265,
            "n_negs": 1
        },
        "optim": {
            "pipeline": "CRA",
            "epochs": 20,
            "lr": 0.0001,
            "warmup_proportion": 0.1,
            "batch_size": 32,
            "save_period": 1,
            "print_iter": 100
        },
        "stat": {
            "monitor": {
                "mode": "max",
                "metric": "Acc"
            },
            "early_stop": 10
        },
        "misc": {
            "cuda": "0",
            "seed": 42,
            "result_dir": "results/vqa_quickstart"
        }
    }
    
    
    print("CausalVLR VQA Quick Start")
    print("=" * 60)

    # Create Pipeline 
    print("\n Creating CRA Pipeline...")
    pipeline = CRAPipeline(config)
    print(f"Model created successfully")
    
    # Training Model 
    print("\n Starting training...")
    print(f"Epochs: {config['optim']['epochs']}")
    print(f"Learning rate: {config['optim']['lr']}")
    print(f"Batch size: {config['dataset']['batch_size']}")
    print(f"Save directory: {config['misc']['result_dir']}")
    
    # Run training
    train_results = pipeline.train()
    
    print(f"\nTraininging complete")
    print(f"Best validation accuracy: {train_results['best_val_acc']:.4f}")
    
    # Evaluate Model 
    print("\n Evaluating on test set...")
    test_results = pipeline.inference()
    
    # Print results
    print("\n" + "=" * 60)
    print("Testing Results")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    
    if 'per_type_acc' in test_results:
        print("\nAccuracy by question type:")
        for qtype, acc in test_results['per_type_acc'].items():
            print(f"  {qtype:15s}: {acc:.4f}")
    
    # Save results
    output_file = f"{config['misc']['result_dir']}/test_results.yaml"
    with open(output_file, 'w') as f:
        yaml.dump(test_results, f)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    # Check CUDA
    if not torch.cuda.is_available():
        print("Warning: CUDA not detected, will use CPU for training (slower)")
    else:
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    
    main()
