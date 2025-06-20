---
layout: archive
title: "MOOSE"
permalink: /MOOSE/
author_profile: true
---
---
## 🦌 (MOOSE) Mixture of Ordered Scoring Experts for Cross-prompt Essay Trait Scoring 🦌

<div align="center" >
This is synopsis page of <strong>Mixture of Ordered Scoring Experts for Cross-prompt Essay Trait Scoring</strong> which is accepted to ACL 2025!
</div>

<div align="center">

  ## Model overview 
  <img src="/images/aes.png"  width="60%" height="60%"/>
</div>

**MOOSE** framework is **the-state-of-the-art** model on **multi-trait cross prompt essay scoring**. It consist of serveral mechanism:

1. **Multi Chunk content feature extractor**: Extract content features of the essay and prompt ,with multi-granularity, via document BERT and multi-chunk BERT models which is follow [(Wang et al., 2022)](https://aclanthology.org/2022.naacl-main.249.pdf). The content features are composed of chuncks at different levels of granularity, enabling the model not only to analyze the essay as a whole, but also to capture the performance of different segments within the essay.

2. **Multi-trait attention**: adopt the architecture of trait attention mechanism proposed in ProTACT [(Do et al.,2023)](https://aclanthology.org/2023.findings-acl.98.pdf) to learn trait-specific representations. Different from the ProTACT, we use the document BERT feature as the query and the multi-chunk BERT features as both key and value in the cross-attention
layer to learn non-prompt specific representation of the essay. 

3. **Mixture of Ordered Scoring Experts**: a mechanism designed to mimic human expert reasoning through a three-stage Ordered Scoring Experts (OSE) framework: a scoring expert learns trait-specific signals from essay features, a ranking expert adaptively selects scoring cues via a Mixture-of-Experts (MoE) mechanism, and an adherence expert assesses the alignment between essay content and prompt. The model employs a dual-layer cross-attention decoder, with stop-gradient applied to queries to simulate scoring cue retrieval. Together, these components enable more accurate and robust essay trait evaluation.
<div align="center">
  <img src="/images/moose.png"  width="60%" height="60%"/>
</div>



## Source Code
* The source code of the model is [here](https://github.com/antslabtw/MOOSE-AES)
* The online scoring engine is [here](https://github.com/tempxdxd)
## Citation
To be update
