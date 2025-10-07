def build_tokenizer(config, task=None):
    from causalvlr.utils.MRG import tokenizers_fn as mrg_tokenizers
    from causalvlr.data.VQA import build_tokenizer as vqa_build_tokenizer
    
    if task is None:
        task = _detect_task(config)
    
    task = task.upper()
    
    if task == 'MRG':
        tokenizer_name = config.get('tokenizer', 'ori')
        return mrg_tokenizers[tokenizer_name](config)
    elif task == 'VQA':
        return vqa_build_tokenizer(config)
    else:
        raise ValueError(f"Unknown task type: {task}")


def _detect_task(config):
    if 'task' in config and config['task'] in ['finetune', 'pretrain', 'inference']:
        return 'MRG'
    
    if 'dataset' in config:
        if isinstance(config['dataset'], dict) and 'name' in config['dataset']:
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
    
    raise ValueError("Cannot determine task type from config")


__all__ = ['build_tokenizer']
