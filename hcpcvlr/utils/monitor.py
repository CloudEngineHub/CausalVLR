import numpy as np


class Monitor(object):
    """
    statistic level:
        level 1 (defalut): loss and metric
        level 2: 
    """
    def __init__(self, cfgs) -> None:
        self.cfgs = cfgs
        self.statistic_level = cfgs["statistic_level"] 
        self.recorder = {}
    
    def addInfo(self, info: dict):
        for key in info.keys():
            self.recorder[key] = info[key]

