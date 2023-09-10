"""
pipeline for reasoning via LLM
"""
import pandas as pd
from .base import BasePipeline
import numpy as np
import torch
import random
import time
import os

class ChatPipeline(BasePipeline):
    def __init__(self, model, cfgs, 
            metric_caculator=None, 
            test_dataloader=None
    ):
        random.seed(cfgs["seed"])
        np.random.seed(cfgs["seed"])
        super(ChatPipeline, self).__init__(model, cfgs)
        self.test_dataloader = test_dataloader
        self.metric_caculator = metric_caculator

    def _train_epoch(self, epoch):
        pass

    def inference(self):
        """
        This function takes an integer and a string as input and returns a boolean.
        
        Args:
            loader: 
        
        Returns:
            dict: the score of each metric
        """
        problems, qids, shot_qids = self.test_dataloader(self.cfgs)
        _qid, _answer, _output = [], [], []
        for i in range(len(qids)):
            qid = qids[i]
            choice = problems[qid]["choices"]
            prompt = self.model.build_prompt_sciqa(problems, qid)
            qid, answer, output = self.model(qid, prompt, choice)
            _qid.append(qid)
            _answer.append(answer)
            _output.append(output)
        return _qid, _answer, _output
