# UniT: Unified Geometry Learning with Group Autoregressive Transformer

<p align="center">
  <a href="https://wang-xjtu.github.io/">Haotian Wang</a><sup>1</sup>,
  Yusong Huang<sup>1</sup>,
  Zhaonian Kuang<sup>2,1</sup>,
  Hongliang Lu<sup>1</sup>,
  <a href="https://facultyprofiles.hkust-gz.edu.cn/faculty-personal-page?id=168">Xinhu Zheng</a><sup>1,†</sup>,
  <a href="http://www.lianpp.com/xjtu/mu_gr/web/mengyang">Meng Yang</a><sup>2,†</sup>,
  and <a href="https://www.ganghua.org/">Gang Hua</a><sup>3</sup>
</p>

<p align="center">
  <sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou)&nbsp;&nbsp;
  <sup>2</sup>Xi'an Jiaotong University&nbsp;&nbsp;
  <sup>3</sup>Amazon.com, Inc.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.21131"><img src="https://img.shields.io/badge/arXiv-UniT-red" alt="Paper PDF"></a>
  <a href="https://sc2i-hkustgz.github.io/UniT/"><img src="https://img.shields.io/badge/Project_Page-UniT-green" alt="Project Page"></a>
  <a href="https://enceladush-unit.hf.space/"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue" alt="Hugging Face Demo"></a>
</p>

<!-- Replace the remaining placeholder href="#" above with the arXiv URL. -->

## News

- **2026-05-21:** Paper, project page, and Hugging Face demo are released.

## Overview

<p align="center">
  <img src="assets/teaser.png" alt="UniT teaser" width="100%">
</p>

UniT is a unified feed-forward model that reformulates a wide range of geometry perception capabilities into a single framework, covering diverse *view configurations*, *modality combinations*, *metric-scale perception*, and *long-horizon scalability*. It supports both online and offline inference over an arbitrary number of views, flexibly incorporates auxiliary modalities such as camera parameters and depth maps, recovers geometry in metric scale measured in meters, and maintains bounded complexity over long horizons in in-the-wild environments.

## Availability

The paper is currently under review, and the code is not publicly available at this stage. In the meantime, we provide a Hugging Face demo for testing UniT.

## Citation

If you find UniT useful in your research, please consider citing:

```bibtex
@misc{wang2026unit,
      title={UniT: Unified Geometry Learning with Group Autoregressive Transformer}, 
      author={Haotian Wang and Yusong Huang and Zhaonian Kuang and Hongliang Lu and Xinhu Zheng and Meng Yang and Gang Hua},
      year={2026},
      eprint={2605.21131},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.21131}, 
}
```
