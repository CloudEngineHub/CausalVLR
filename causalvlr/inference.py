from causalvlr.api.pipeline.MRG import MRGPipeline
from causalvlr.api.pipeline.VQA import CRAPipeline, TempCLIPPipeline


def inference(config, checkpoint_path=None, task=None):
    if task is None:
        task = _detect_task(config)
    
    task = task.upper()
    
    if task == 'MRG':
        if checkpoint_path:
            config["load_model_path"] = checkpoint_path
        pipeline = MRGPipeline(config)
        return pipeline.inference()
    elif task == 'VQA':
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
            pipeline = CRAPipeline(config)
        elif pipeline_name == 'TEMPCLIP':
            pipeline = TempCLIPPipeline(config)
        else:
            raise ValueError(f"Unknown VQA pipeline: {pipeline_name}")
        
        if checkpoint_path:
            pipeline._resume_checkpoint(checkpoint_path)
        
        return pipeline.infer()
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
    
    if 'optim' in config and 'pipeline' in config['optim']:
        pipeline_name = config['optim']['pipeline']
        if pipeline_name in ['CRA', 'TempCLIP']:
            return 'VQA'
    
    raise ValueError("Cannot determine task type from config")


__all__ = ['inference']
