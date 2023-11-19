# Awesome-LLMs-in-Graph-tasks
![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg) ![GitHub stars](https://img.shields.io/github/stars/yhLeeee/Awesome-LLMs-in-Graph-tasks.svg)
> A collection of papers on **Large Language Models in Graph Tasks**. We will try to make this list updated frequently. If you found any error or any missed paper, please don't hesitate to open issues or pull requests.

## How can LLMs help improve graph-related tasks?

With the help of LLMs, there has been a notable shift in the way we interact with graphs, particularly those containing nodes associated with text attributes. The integration of LLMs with traditional GNNs can be mutually beneficial and enhance graph learning. While GNNs are proficient at capturing structural information, they primarily rely on semantically constrained embeddings as node features, limiting their ability to express the full complexities of the nodes. Incorporating LLMs, GNNs can be enhanced with stronger node features that effectively capture both structural and contextual aspects. On the other hand, LLMs excel at encoding text but often struggle to capture structural information present in graph data. Combining GNNs with LLMs can leverage the robust textual understanding of LLMs while harnessing GNNs' ability to capture structural relationships, leading to more comprehensive and powerful graph learning.

<p align="center"><img src="Figures/overview.png" width=75% height=75%></p>
<p align="center"><em>Figure 1.</em> The overview of Graph Meets LLMs.</p>

## Table of Contents

- [Awesome-LLMs-in-Graph-tasks](#awesome-llms-in-graph-tasks)
  - [How can LLMs help improve graph-related tasks](#how-can-llms-help-improve-graph-related-tasks)
  - [Table of Contents](#table-of-contents)
  - [LLM as Enhancer](#llm-as-enhancer)
  - [LLM as Predictor](#llm-as-predictor)
  - [GNN-LLM Alignment](#gnn-llm-alignment)
  - [Others](#others)
  - [Contributing](#contributing)


## LLM as Enhancer
* (_2022.03_) [ICLR‘ 2022] **Node Feature Extraction by Self-Supervised Multi-scale Neighborhood Prediction** [[Paper](https://arxiv.org/abs/2111.00064) | [Code](https://github.com/amzn/pecos/tree/mainline/examples/giant-xrt)]
   <details close>
   <summary>GIANT</summary>
   <p align="center"><img width="75%" src="Figures/GIANT.jpg" /></p>
   <p align="center"><em>The framework of GIANT.</em></p>
   </details>
* (_2023.02_) [ICLR' 2023] **Edgeformers: Graph-Empowered Transformers for Representation Learning on Textual-Edge Networks** [[Paper](https://arxiv.org/abs/2302.11050) | [Code](https://github.com/PeterGriffinJin/Edgeformers)]
   <details close>
   <summary>Edgeformers</summary>
   <p align="center"><img width="75%" src="Figures/Edgeformers.jpg" /></p>
   <p align="center"><em>The framework of Edgeformers.</em></p>
   </details>
* (_2023.05_) [KDD' 2023] **Graph-Aware Language Model Pre-Training on a Large Graph Corpus Can Help Multiple Graph Applications** [[Paper](https://arxiv.org/abs/2306.02592)]
   <details close>
   <summary>GALM</summary>
   <p align="center"><img width="75%" src="Figures/GALM.jpg" /></p>
   <p align="center"><em>The framework of GALM.</em></p>
   </details>
* (_2023.06_) [KDD' 2023] **Heterformer: Transformer-based Deep Node Representation Learning on Heterogeneous Text-Rich Networks** [[Paper](https://dl.acm.org/doi/abs/10.1145/3580305.3599376?casa_token=M9bG1HLyTEYAAAAA:gIiYO9atgtxNaBgfKpy4D3N66QDkCFLFvlEADvzC8Pobe_EWausOknGnRFzdDF-Xnq-vbWAWMT1qkA) | [Code](https://github.com/PeterGriffinJin/Heterformer)]
   <details close>
   <summary>Heterformer</summary>
   <p align="center"><img width="75%" src="Figures/Heterformers.jpg" /></p>
   <p align="center"><em>The framework of Heterformer.</em></p>
   </details>
* (_2023.05_) [Arxiv' 2023] **Harnessing Explanations: LLM-to-LM Interpreter for Enhanced Text-Attributed Graph Representation Learning** [[Paper](https://arxiv.org/abs/2305.19523) | [Code](https://github.com/XiaoxinHe/TAPE)]
   <details close>
   <summary>TAPE</summary>
   <p align="center"><img width="75%" src="Figures/TAPE.jpg" /></p>
   <p align="center"><em>The framework of TAPE.</em></p>
   </details>
* (_2023.08_) [Arxiv' 2023] **Exploring the potential of large language models (llms) in learning on graphs** [[Paper](https://arxiv.org/abs/2307.03393)]
   <details close>
   <summary>KEA</summary>
   <p align="center"><img width="75%" src="Figures/KEA.jpg" /></p>
   <p align="center"><em>The framework of KEA.</em></p>
   </details>
* (_2023.07_) [Arxiv' 2023] **Can Large Language Models Empower Molecular Property Prediction?** [[Paper](https://arxiv.org/abs/2307.07443) | [Code](https://github.com/ChnQ/LLM4Mol)]
   <details close>
   <summary>LLM4Mol</summary>
   <p align="center"><img width="75%" src="Figures/LLM4Mol.jpg" /></p>
   <p align="center"><em>The framework of LLM4Mol.</em></p>
   </details>
* (_2023.08_) [Arxiv' 2023] **Simteg: A frustratingly simple approach improves textual graph learning** [[Paper](https://arxiv.org/abs/2308.02565) | [Code](https://github.com/vermouthdky/SimTeG)]
   <details close>
   <summary>SimTeG</summary>
   <p align="center"><img width="75%" src="Figures/SimTeG.jpg" /></p>
   <p align="center"><em>The framework of SimTeG.</em></p>
   </details>
* (_2023.09_) [Arxiv' 2023] **Prompt-based Node Feature Extractor for Few-shot Learning on Text-Attributed Graphs** [[Paper](https://arxiv.org/abs/2309.02848)]
   <details close>
   <summary>G-Prompt</summary>
   <p align="center"><img width="75%" src="Figures/G-Prompt.jpg" /></p>
   <p align="center"><em>The framework of G-Prompt.</em></p>
   </details>
* (_2023.09_) [Arxiv' 2023] **TouchUp-G: Improving Feature Representation through Graph-Centric Finetuning** [[Paper](https://arxiv.org/abs/2309.13885)]
   <details close>
   <summary>TouchUp-G</summary>
   <p align="center"><img width="75%" src="Figures/TouchUp-G.jpg" /></p>
   <p align="center"><em>The framework of TouchUp-G.</em></p>
   </details>
* (_2023.09_) [Arxiv' 2023] **One for All: Towards Training One Graph Model for All Classification Tasks** [[Paper](https://arxiv.org/abs/2310.00149) | [Code](https://github.com/LechengKong/OneForAll)]
   <details close>
   <summary>OFA</summary>
   <p align="center"><img width="75%" src="Figures/OFA.jpg" /></p>
   <p align="center"><em>The framework of OFA.</em></p>
   </details>
* (_2023.11_) [WSDM' 2023] **LLMRec: Large Language Models with Graph Augmentation for Recommendation** [[Paper](https://arxiv.org/abs/2311.00423) | [Code](https://github.com/HKUDS/LLMRec)]
   <details close>
   <summary>LLMRec</summary>
   <p align="center"><img width="75%" src="Figures/LLMRec.jpg" /></p>
   <p align="center"><em>The framework of LLMRec.</em></p>
   </details>
* (_2023.11_) [NeurIPS' 2023] **WalkLM: A Uniform Language Model Fine-tuning Framework for Attributed Graph Embedding** [[Paper](https://openreview.net/forum?id=ZrG8kTbt70) | [Code](https://github.com/Melinda315/WalkLM)]
   <details close>
   <summary>WalkLM</summary>
   <p align="center"><img width="75%" src="Figures/WalkLM.jpg" /></p>
   <p align="center"><em>The framework of WalkLM.</em></p>
   </details>
* (_2023.10_) [Arxiv' 2023] **Learning Multiplex Embeddings on Text-rich Networks with One Text Encoder** [[Paper](https://arxiv.org/abs/2310.06684) | [Code](https://github.com/PeterGriffinJin/METERN-submit)]
   <details close>
   <summary>METERN</summary>
   <p align="center"><img width="75%" src="Figures/METERN.jpg" /></p>
   <p align="center"><em>The framework of METERN.</em></p>
   </details>

## LLM as Predictor


## GNN-LLM Alignment


## Others

## Contributing
If you have come across relevant resources, feel free to open an issue or submit a pull request.

```
* (_time_) [conference] **paper_name** [[Paper](link) | [Code](link)]
   <details close>
   <summary>Model name</summary>
   <p align="center"><img width="75%" src="Figures/xxx.jpg" /></p>
   <p align="center"><em>The framework of model name.</em></p>
   </details>
```
