# OPTP: Orthogonal Projection Token Pruning for Efficient Multi-modal Large Language Models
[[`Paper`] | [`📂Logs`](https://drive.google.com/drive/folders/1pat-szhxEG6DW6rtiosysZL2eKOTRsOC?usp=sharing)]




> **Abstract**:
Visual token pruning has recently emerged as a indispensable technique for enabling efficient inference in Multi-modal Large Language Models (MLLMs), where the rapidly increasing number of visual tokens incurs substantial computational overhead.However, existing methods predominantly rely on heuristic saliency estimation or distance-based diversity metrics, making them susceptible to the feature coupling effect, leading to either severe inter-token semantic overlap or the retention of task-irrelevant visual features.To this end, we propose OPTP Orthogonal Projection Token Pruning), a training-free geometric framework that reformulates visual token selection as a dynamic residual extraction process through iterative orthogonal projection, jointly preserving semantic saliency and feature diversity.Specifically, OPTP iteratively projects candidate tokens onto the orthogonal complement of the subspace spanned by previously selected tokens, explicitly removing redundant semantic components while promoting complementary token selection according to their marginal information gain, thereby producing a compact yet highly informative visual representation for multi-modal reasoning.Extensive experiments on multiple benchmarks demonstrate that OPTP achieves near-lossless performance with only 33.3% token retention, preserving 99.42% of the original performance while providing a training-free and plug-and-play solution for efficient MLLM inference.
## News
- [2026.7.24] We release the code of OPTP for LLaVA.


## Installation
1. Install the environment of [LLaVA](https://github.com/haotian-liu/LLaVA).
```
conda create -n optp python=3.10 -y
conda activate optp
pip install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 

git clone git@github.com:lvzhang11112/OPTP-pruning.git
cd SCOPE

pip install -r requirements.txt
cd LLaVA
pip install -e .
cd ..
```
2. Install our OPTP method by running the following command:
```
pip install -e .
```

## Usage
```
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from scope import scope
model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
## 64 tokens are retained
model = optp(model, token_num=64)
```

## Main Results

1. Results on LLaVA 1.5 7B with 64 tokens:
```
bash run_llava.sh 64
```

2. Results on LLaVA-Next 7B with 160 tokens:
```
bash run-llava-next.sh 160
```




## Acknowledgement
- This work is built upon [LLaVA](https://llava-vl.github.io/), [Lmms-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). We thank them for their excellent open-source contributions.

- We also thank [VisionZip](https://github.com/dvlab-research/VisionZip), [DivPrune](https://github.com/vbdi/divprune), [FastV](https://github.com/pkunlp-icler/FastV), [SparseVLM](https://github.com/Gumpest/SparseVLMs), and others for their contributions, which have provided valuable insights.

## Citation

```
