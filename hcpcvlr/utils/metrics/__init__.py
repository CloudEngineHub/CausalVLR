from .nlg.bleu.bleu import Bleu
from .nlg.meteor import Meteor
from .nlg.rouge import Rouge
from .nlg.cider.cider import Cider


METRIC_LIST = {
    # NLG
    "Meteor": Meteor,
    "Rouge": Rouge,
    "Cider": Cider,
    "Bleu": Bleu
}


class MetricCalculator():
    def __init__(self, cfgs):
        # TODO
        self.metric_name = cfgs["metric"]
    
    def get_scorer(self):
        scorers = []
        for key in self.metric_name:
            if key == "Bleu":
                scorers.append((Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]))    
            else:
                scorers.append((METRIC_LIST[key](), key))
        return scorers
    
    def compute_scores(self, gts, res):
        """
        Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

        :param gts: Dictionary with the image ids and their gold captions,
        :param res: Dictionary with the image ids ant their generated captions
        :print: Evaluation score (the mean of the scores of all the instances) for each measure
        """
        # Set up scorers
        scorers = self.get_scorer()
        eval_res = {}
        # Compute score for each metric
        for scorer, method in scorers:
            try:
                score, scores = scorer.compute_score(gts, res, verbose=0)
            except TypeError:
                score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, m in zip(score, method):
                    eval_res[m] = sc
            else:
                eval_res[method] = score
        return eval_res
