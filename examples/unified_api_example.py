"""
Inference using Unified API
Unified API Inference Example

Demonstrates how to use CausalVLR's unified API interface for inference.
"""

import json
import yaml
from causalvlr import build_tokenizer, inference


def mrg_inference_example():
    print("MRG Inference Example")
    
    # Load configuration
    with open('configs/MRG/vlci.json', 'r') as f:
        config = json.load(f)
    
    # Set checkpoint path
    checkpoint_path = 'results/mrg_vlci/best_model.pth'
    
    # Method 1: Use unified inference interface
    print("\n[Method 1] Using unified inference interface...")
    results = inference(
        config=config,
        checkpoint_path=checkpoint_path,
        task='MRG'  # Optional, will auto-detect
    )
    
    print(f"Generated count: {len(results['predictions'])}")
    
    # Print metrics
    print("\nEvaluation Metrics:")
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Show examples
    print("\nGenerated Examples:")
    for i in range(min(3, len(results['predictions']))):
        print(f"\nExample {i+1}:")
        print(f"  Ground truth: {results['ground_truth'][i]}")
        print(f"  Generated: {results['predictions'][i]}")
    
    return results


def vqa_inference_example():
    print("VQA Inference Example")
    # Load configuration
    with open('configs/VQA/CRA/CRA_NextGQA.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set checkpoint path
    checkpoint_path = 'results/vqa_cra/best_model.pth'
    
    # Method 1: Use unified inference interface
    print("\n[Method 1] Using unified inference interface...")
    results = inference(
        config=config,
        checkpoint_path=checkpoint_path,
        task='VQA'  # Optional, will auto-detect
    )
    
    print(f"Inference complete")
    print(f"Accuracy: {results['accuracy']:.4f}")
    
    # Print accuracy by type
    if 'per_type_acc' in results:
        print("\nAccuracy by question type:")
        for qtype, acc in results['per_type_acc'].items():
            print(f"  {qtype}: {acc:.4f}")
    
    return results


def tokenizer_example():
    print("Tokenizer Example")
    
    # MRG Tokenizer
    print("\nMRG Tokenizer:")
    with open('configs/MRG/vlci.json', 'r') as f:
        mrg_config = json.load(f)
    
    mrg_tokenizer = build_tokenizer(mrg_config, task='MRG')
    
    text = "Findings: No acute cardiopulmonary process."
    tokens = mrg_tokenizer.encode(text)
    decoded = mrg_tokenizer.decode(tokens)
    
    print(f"  Original: {text}")
    print(f"  Token IDs: {tokens[:10]}...")
    print(f"  Decoded: {decoded}")
    
    # VQA Tokenizer
    print("\nVQA Tokenizer:")
    with open('configs/VQA/CRA/CRA_NextGQA.yml', 'r') as f:
        vqa_config = yaml.safe_load(f)
    
    vqa_tokenizer = build_tokenizer(vqa_config, task='VQA')
    
    question = "What is the person doing in the video?"
    encoded = vqa_tokenizer(
        question,
        padding='max_length',
        max_length=30,
        truncation=True,
        return_tensors='pt'
    )
    
    print(f"  Question: {question}")
    print(f"  Input IDs shape: {encoded['input_ids'].shape}")
    print(f"  Attention Mask shape: {encoded['attention_mask'].shape}")


def batch_inference_example():
    print("Batch Inference Example")
    
    from pathlib import Path
    from PIL import Image
    import torch
    from causalvlr.api.pipeline.MRG import MRGPipeline
    
    # Load configuration and model
    with open('configs/MRG/vlci.json', 'r') as f:
        config = json.load(f)
    
    config['train']['load_model_path'] = 'results/mrg_vlci/best_model.pth'
    pipeline = MRGPipeline(config)
    pipeline.model.eval()
    
    # Get test images
    image_dir = Path('data/test_images')
    image_paths = list(image_dir.glob('*.png'))[:10]  # Testing 10 images
    
    print(f"\nProcessing {len(image_paths)}  images...")
    
    results = []
    for img_path in image_paths:
        # Load image
        image = Image.open(img_path).convert('RGB')
        image_tensor = pipeline.transform(image).unsqueeze(0).cuda()
        
        # Generated
        with torch.no_grad():
            report = pipeline.model.generate(image_tensor, pipeline.tokenizer)
        
        results.append({
            'image_id': img_path.stem,
            'report': report
        })
        
        print(f"  {img_path.name}: {report[:50]}...")
    
    # Save results
    with open('batch_inference_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: batch_inference_results.json")
    
    return results


def main():
    print("CausalVLR Unified API Examples")
    print("=" * 60)
    
    # MRG Inference
    try:
        mrg_results = mrg_inference_example()
    except Exception as e:
        print(f"MRG inference failed: {e}")
    
    # VQA Inference
    try:
        vqa_results = vqa_inference_example()
    except Exception as e:
        print(f"VQA inference failed: {e}")
    
    # Tokenizer Usage
    try:
        tokenizer_example()
    except Exception as e:
        print(f"Tokenizer example failed: {e}")
    
    # Batch Inference
    try:
        batch_results = batch_inference_example()
    except Exception as e:
        print(f"Batch inference failed: {e}")
    
    print("All examples complete")


if __name__ == "__main__":
    main()
