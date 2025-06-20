
# Mixture of Ordered Scoring Experts for Cross-prompt Essay Trait Scoring (MOOSE)

This repository contains the code used to produce the results from the paper Automated Cross-prompt Scoring of Essay Traits published in [ACL 2025](https:/acl.org/moose).

## Model overview

<div align="center">
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

## Package Requirements

Install below packages in your virtual environment before running the code.
- python==3.7.11
- tensorflow=2.0.0
- numpy=1.18.1
- nltk=3.4.5
- pandas=1.0.5
- scikit-learn=0.22.1

## How to Run MOOSE
This bash script will run each model 5 times with different seeds ([12, 22, 32, 42, 52]).
- `bash ./train_ProTACT.sh`

\* Topic-coherence features are included in the `data/LDA/hand_crafted_final_{prompt}.csv` file as the 'highest_topic' column.

\* Note that every run does not produce the same results due to the random elements.

## Note – Handcrafted Features:
We utilize features follow [CTS (Ridley et al,2021)](https://github.com/robert1ridley/cross-prompt-trait-scoring/tree/main). Same as (Ridley et al,2021), the handcrafted features have been precomputed and are available at data/hand_crafted_v3.csv. Additional readability-related features can be found in data/allreadability.pickle. The scripts used to generate these features are features.py and create_readability_features.py, respectively.

If you wish to regenerate the features, make sure to install the following Python packages:

textstat

spacy (along with the English model: python -m spacy download en_core_web_sm)

readability (install via: pip install https://github.com/andreasvc/readability/tarball/master)

## demo link
* The online scoring engine is [here](https://github.com/tempxdxd)
## Citation
To be update
