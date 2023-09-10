import sys
import os.path as osp
sys.path.append(osp.join(osp.dirname(__file__), '../hcpcvlr'))

from utils.cfgs_loader import load_yaml
from api.pipeline import ChatPipeline
from utils.metrics import MetricCalculator
from data import load_scienceqa_data
from models.chat.caco_cot import CaCoCoT


cfgs = load_yaml("configs/chat/CaCo_CoT.yaml")
# print(cfgs)
model = CaCoCoT(cfgs)
# -------------------
# inference on Science QA
# -------------------
work = ChatPipeline(model, cfgs, metric_caculator=MetricCalculator(cfgs))
work.inference()
