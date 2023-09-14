# Towards CausalGPT: A Multi-Agent Approach for Faithful Knowledge Reasoning via Promoting Causal Consistency in LLMs

<div align=center>

![Image](CaCo_demo.gif)

## |[📜 paper](https://arxiv.org/pdf/2308.11914.pdf) | [🌌 code](https://github.com/RndmVariableQ/CaCo-CoT) |

</div>

---
Despite advancements in LLMs, knowledge-based reasoning remains a longstanding issue due to the fragility of knowledge recall and inference. Existing methods primarily encourage LLMs to autonomously plan and solve problems or to extensively sample reasoning chains without addressing the conceptual and inferential fallacies. Attempting to alleviate inferential fallacies and drawing inspiration from multi-agent collaboration, we present a framework to increase faithfulness and causality for knowledge-based reasoning. Specifically, we propose to employ multiple intelligent agents (i.e., reasoners and an evaluator) to work collaboratively in a reasoning-and-consensus paradigm for elevated reasoning faithfulness. The reasoners focus on providing solutions with human-like causality to solve open-domain problems. On the other hand, the evaluator agent scrutinizes if a solution is deducible from a non-causal perspective and if it still holds when challenged by a counterfactual candidate. According to the extensive and comprehensive evaluations on a variety of knowledge reasoning tasks (e.g., science question answering and commonsense reasoning), our framework outperforms all compared state-of-the-art approaches by large margins.


## Benchmarks

[ScienceQA](https://scienceqa.github.io/), [Com2Sense](https://github.com/PlusLabNLP/Com2Sense), [BoolQ](https://github.com/google-research-datasets/boolean-questions).      

## Citation
```
@misc{tang2023causalgpt,
      title={Towards CausalGPT: A Multi-Agent Approach for Faithful Knowledge Reasoning via Promoting Causal Consistency in LLMs}, 
      author={Ziyi Tang and Ruilin Wang and Weixing Chen and Keze Wang and Yang Liu and Tianshui Chen and Liang Lin},
      year={2023},
      eprint={2308.11914},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```