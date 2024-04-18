from hcpcvlr.utils.cfgs_loader import load_yaml
from hcpcvlr.api.pipeline import ChatPipeline
from hcpcvlr.utils.metrics import MetricCalculator
from hcpcvlr.data import load_scienceqa_data
from hcpcvlr.models.chat.caco_cot import CaCoCoT


cfgs = load_yaml("configs/chat/CaCo_CoT.yaml")
# print(cfgs)
model = CaCoCoT(cfgs)
# -------------------
# inference on Science QA
# -------------------
work = ChatPipeline(model, cfgs, metric_caculator=MetricCalculator(cfgs))
work.inference()
