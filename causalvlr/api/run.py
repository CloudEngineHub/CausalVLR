import argparse
import json
import random
from collections import OrderedDict
import numpy as np
import torch
import yaml
import warnings

warnings.filterwarnings('ignore')


def load_config(config_path):
    ext = config_path.split('.')[-1].lower()
    
    if ext == 'json':
        json_str = ''
        with open(config_path, 'r') as f:
            for line in f:
                line = line.split('//')[0] + '\n'
                json_str += line
        config = json.loads(json_str, object_pairs_hook=OrderedDict)
        dict_args = {}
        for key in config.keys():
            dict_args.update(config[key])
        return dict_args
    elif ext in ['yml', 'yaml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        raise ValueError(f"Unsupported config format: {ext}. Use .json, .yml, or .yaml")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detect_task_type(config):
    if 'task' in config:
        task = config['task']
        if task in ['finetune', 'pretrain', 'inference']:
            return 'MRG'
    
    if 'dataset' in config and 'name' in config['dataset']:
        dataset_name = config['dataset']['name'].lower()
        if dataset_name in ['nextqa', 'star', 'next-qa']:
            return 'VQA'
        elif dataset_name in ['iu_xray', 'mimic_cxr', 'iu-xray', 'mimic-cxr']:
            return 'MRG'
    
    if 'model' in config:
        if isinstance(config['model'], dict) and 'name' in config['model']:
            model_name = config['model']['name'].lower()
            if model_name in ['cra', 'tempclip']:
                return 'VQA'
            elif model_name in ['baseline', 'vlci', 'vlp']:
                return 'MRG'
        elif isinstance(config['model'], str):
            model_name = config['model'].lower()
            if model_name in ['baseline', 'vlci', 'vlp']:
                return 'MRG'
    
    if 'optim' in config and 'pipeline' in config['optim']:
        pipeline_name = config['optim']['pipeline']
        if pipeline_name in ['CRA', 'TempCLIP']:
            return 'VQA'
    
    raise ValueError("Cannot determine task type from config. Please specify task explicitly.")


def create_pipeline(config, task_type=None):
    if task_type is None:
        task_type = detect_task_type(config)
    
    task_type = task_type.upper()
    
    if task_type == 'MRG':
        from causalvlr.api.pipeline.MRG import MRGPipeline
        return MRGPipeline(config)
    elif task_type == 'VQA':
        from causalvlr.api.pipeline.VQA import CRAPipeline, TempCLIPPipeline
        
        pipeline_name = config.get('optim', {}).get('pipeline', None)
        if pipeline_name is None:
            if 'model' in config:
                model_name = config['model'].get('name', '').lower() if isinstance(config['model'], dict) else config['model'].lower()
                if 'cra' in model_name:
                    pipeline_name = 'CRA'
                elif 'tempclip' in model_name:
                    pipeline_name = 'TempCLIP'
        
        # Normalize pipeline name to uppercase
        if pipeline_name:
            pipeline_name = pipeline_name.upper()
        
        if pipeline_name == 'CRA':
            return CRAPipeline(config)
        elif pipeline_name == 'TEMPCLIP':
            return TempCLIPPipeline(config)
        else:
            raise ValueError(f"Unknown VQA pipeline: {pipeline_name}")
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def main():
    parser = argparse.ArgumentParser(description='CausalVLR Unified Pipeline')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to config file (.json, .yml, or .yaml)')
    parser.add_argument('--task', '-t', type=str, default=None, choices=['MRG', 'VQA'],
                        help='Task type (MRG or VQA). Auto-detected if not specified.')
    parser.add_argument('--mode', '-m', type=str, default='train', choices=['train', 'inference', 'infer'],
                        help='Running mode: train or inference')
    parser.add_argument('--cuda', type=str, default=None,
                        help='CUDA device(s) to use')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.cuda is not None:
        if 'misc' in config:
            config['misc']['cuda'] = args.cuda
        elif 'cuda' in config:
            config['cuda'] = args.cuda
    
    if args.seed is not None:
        config['seed'] = args.seed
    
    if 'seed' in config:
        if config['seed'] == -1:
            config['seed'] = np.random.randint(0, 23333)
        setup_seed(config['seed'])
    
    print(f"Config: {args.config}")
    print(f"Task Type: {args.task if args.task else 'Auto-detect'}")
    print(f"Mode: {args.mode}")
    
    pipeline = create_pipeline(config, args.task)
    
    if args.mode in ['inference', 'infer']:
        pipeline.inference()
    else:
        pipeline.train()


if __name__ == '__main__':
    main()
