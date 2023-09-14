# Cross-Modal Causal Intervention for Medical Report Generation

<div align=center>

![Image](vlci_demo.gif)

## |[📜 paper](https://arxiv.org/pdf/2303.09117.pdf) | [🌌 code](https://github.com/WissingChen/VLCI) |

</div>

---



Medical report generation (MRG) is essential for computer-aided diagnosis and medication guidance, which can relieve the heavy burden of radiologists by automatically generating the corresponding medical reports according to the given radiology image. However, due to the spurious correlations within image-text data induced by visual and linguistic biases, it is challenging to generate accurate reports reliably describing lesion areas. Moreover, the cross-modal confounders are usually unobservable and challenging to be eliminated in an explicit way. In this paper, we aim to mitigate the cross-modal data bias for MRG from a new perspective, i.e., cross-modal causal intervention, and propose a novel Visual-Linguistic Causal Intervention (VLCI) framework for MRG, which consists of a visual deconfounding module (VDM) and a linguistic deconfounding module (LDM), to implicitly mitigate the visual-linguistic confounders by causal front-door intervention. Specifically, due to the absence of a generalized semantic extractor, the VDM explores and disentangles the visual confounders from the patch-based local and global features without expensive fine-grained annotations. Simultaneously, due to the lack of knowledge encompassing the entire medicine, the LDM eliminates the linguistic confounders caused by salient visual features and high-frequency context without constructing a terminology database. Extensive experiments on IU-Xray and MIMIC-CXR datasets show that our VLCI significantly outperforms the state-of-the-art MRG methods.
    
## Benchmarks: 
[IU-Xray](https://pubmed.ncbi.nlm.nih.gov/26133894/), [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/#files-panel) 

## Citation    
```
@misc{chen2023crossmodal,
      title={Cross-Modal Causal Intervention for Medical Report Generation}, 
      author={Weixing Chen and Yang Liu and Ce Wang and Jiarui Zhu and Guanbin Li and Liang Lin},
      year={2023},
      eprint={2303.09117},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```