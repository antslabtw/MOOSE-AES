
# Mixture of Ordered Scoring Experts for Cross-prompt Essay Trait Scoring (MOOSE)

* This repository contains the code used to produce the results from the paper Automated Cross-prompt Scoring of Essay Traits published in [ACL 2025](https:/acl.org/moose). The main page of Moose is [here](https://antslabtw.github.io/MOOSE/) and the online scoring engine is [here](https://github.com/antslabtw/MOOSE-AES)
<br><br>
* MOOSE is multi traits cross prompt essay scoring model which imitates how human raters evaluate essays. MOOSE is composed of three experts:<br>
1) Scoring Expert - Learn essay inherent scoring cues.<br>
2) Ranking Expert - Compare relative quality across different essays.<br>
3) Adherence Expert - Estimate the degree of prompt adherence.<br><br>

## Model overview
<div align="center">
  <img src="/images/moose_illustration.png"  width="20%" height="20%"/>
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

## Requirements

Python version is 3.10.0 

-`pip install -r requirement.txt`

## How to Run MOOSE

Training stage

- `./run_bash.sh`

\* Training, Testing and Validation data in [here](https://drive.google.com/file/d/1csQocg3Kf8yEKX6TUHCSKT983myiUH07/view)

\* After downloading, please extract the contents of the archive to the root folder of the project.

\* or you can use preprocess code which in pre_process folder and download [cross_prompt_attribute.tar.gz](https://drive.google.com/drive/folders/1JD6hj_ml1pWMi572UVa3nSje_7szMKXE) which is follow ProTACT [(Do et al.,2023)](https://aclanthology.org/2023.findings-acl.98.pdf) data partition in the pre_process folder to generate data input. 

\* Note the encode.py is followed Multi-Scale-BERT-AES  [(Wang et al., 2022)](https://github.com/lingochamp/Multi-Scale-BERT-AES/blob/main/encoder.py)

- `cd pre_process`
- `python process.py`


Inference stage

- `./inference.sh`

## Note – Handcrafted Features:
We utilize features follow [CTS (Ridley et al,2021)](https://github.com/robert1ridley/cross-prompt-trait-scoring/tree/main). Same as (Ridley et al,2021), the handcrafted features have been precomputed and are available at prep_process/hand_crafted_v3.csv. Additional readability-related features can be found in pre_process/allreadability.pickle. The scripts used to generate these features are features.py and create_readability_features.py, respectively.

If you wish to regenerate the features, make sure to install the following Python packages:

textstat

spacy (along with the English model: python -m spacy download en_core_web_sm)

readability (install via: pip install https://github.com/andreasvc/readability/tarball/master)

## Citation
<pre style="background: #f6f8fa; padding: 16px; border-radius: 6px; overflow-x: auto;">
<code>
@inproceedings{chen2025mixture,
  title={Mixture of Ordered Scoring Experts for Cross-prompt Essay Trait Scoring},
  author={Chen, Po-Kai and Tsai, Bo-Wei and Wei, Shao Kuan and Wang, Chien-Yao and Wang, Jia-Ching and Huang, Yi-Ting},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={18071--18084},
  year={2025}
}
</code>
</pre>
