# Method
This section provides a summary of representative state-of-the-art (SOTA) algorithms for several Visual-Linguistic Reasoning task, such as visual question answering (VQA) and medical report generation tasks. All algorithms have been implemented using PyTorch. The CausalVLR library will be continuously updated in the coming years. In this section, we will provide a concise introduction to the selected algorithms.

## Update News

### 🔥 **2023.8.19**.
- **v0.0.2** was released in 8/19/2023
- Support [**CaCo-CoT**](../projects/CaCo-CoT/readme.md) for Faithful Reasoning task in LLMs

### 🔥 **2023.6.29**.
- **v0.0.1** was released in 6/30/2023
- Support [**VLCI**](../projects/VLCI/readme.md) for Medical Report Generation task
- Support [**CAMDA (T-PAMI 2023)**](https://github.com/HCPLab-SYSU/CAMDA) for Causality-Aware Medical Diagnosis task
- Support [**CMCIR (T-PAMI 2023)**](../projects/CMCIR/readme.md) for Event-Level Visual Question Answering task
- Support [**VCSR (ACM MM 2023)**](../projects/VCSR/readme.md) for Video Question Answering task
- Support [**Robust Fine-tuning (CVPR 2023)**](../projects/RobustFinetuning/readme.md) for Model Generalization and Robustness

## Model Zoo

<div align="center">

|Task | Model | Benchmark |
| --- | ----- | --------- |
| Medical Report Generation |  VLCI     |    [IU-Xray](https://pubmed.ncbi.nlm.nih.gov/26133894/), [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/#files-panel)       |
| VQA |  CMCIR     |  [SUTD-TrafficQA](https://sutdcv.github.io/SUTD-TrafficQA/#/), [TGIF-QA](https://github.com/YunseokJANG/tgif-qa), [MSVD-QA](https://github.com/xudejing/video-question-answering), [MSRVTT-QA](https://github.com/xudejing/video-question-answering)        |
| Visual Causal Scene Discovery |  VCSR     |    [NExT-QA](https://github.com/doc-doc/NExT-QA), [Causal-VidQA](https://github.com/bcmi/Causal-VidQA), and [MSRVTT-QA](https://github.com/xudejing/video-question-answering)       |
| Model Generalization and Robustness |  Robust Fine-tuning     |    ImageNet-V2, ImageNet-R, ImageNet-Sketch, ObjectNet, ImageNet-A      |
| Causality-Aware Medical Diagnosis |  CAMDA     | [MuZhi](https://aclanthology.org/P18-2033.pdf), [DingXiang](https://github.com/fantasySE/Dialogue-System-for-Automatic-Diagnosis)        |
| Faithful Reasoning in LLMs |  CaCo-CoT     | [ScienceQA](https://scienceqa.github.io/), [Com2Sense](https://github.com/PlusLabNLP/Com2Sense), [BoolQ](https://github.com/google-research-datasets/boolean-questions)|
</div>

## Ongoing Projects