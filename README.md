<div align="center">

# UniT

**Unified Geometry Learning with Group Causal Transformer**

Haotian Wang · Yusong Huang · Zhaonian Kuang · Hongliang Lu<br>
Xinhu Zheng · Meng Yang · Gang Hua

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2605.21131)
[![Project Page](https://img.shields.io/badge/Project-Page-2563eb)](https://omega-hkustgz.github.io/projects/unit/)
[![Hugging Face Demo](https://img.shields.io/badge/Hugging%20Face-Demo-ffcc4d)](https://enceladush-unit.hf.space/)

[Installation](#installation) · [Usage](#usage) · [Outputs](#outputs) · [Visualization](#visualization)

</div>

**UniT** is a feed-forward model for unified geometry learning.

- **Any view configuration:** online, multi-view streaming, and offline inference from one or more views.
- **Flexible modality combinations:** RGB images with optional depth maps and camera parameters.
- **Metric-scale perception:** 3D geometry predicted in meters.
- **Long-horizon scalability:** long-sequence inference with bounded memory cost.

## News

- **2026-11-18:** Inference code and pretrained weights released.
- **2026-05-21:** Paper, project page, and Hugging Face demo released.

## Installation

Use **Python 3.11**:

```bash
git clone https://github.com/Wang-xjtu/UniT.git
cd UniT
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

**Weights:** [Haotian-sx/UniT](https://huggingface.co/Haotian-sx/UniT). The online loader downloads `checkpoint.pt` automatically on first use.

## Usage

### RGB reconstruction

```python
from unit import UniT

model = UniT.from_pretrained("Haotian-sx/UniT")
inputs = {"images": "examples/HKUST_Toy/images"}
predictions = model.predict(inputs)
print(predictions["world_points"].shape)
```

If you [downloaded the checkpoint](https://huggingface.co/Haotian-sx/UniT/resolve/main/checkpoint.pt), load the file directly:

```python
model = UniT.from_checkpoint("checkpoints/checkpoint.pt")
```

`images` accepts a directory (sorted by filename) or an ordered list of paths, PIL images, or RGB NumPy arrays (`[H, W, 3]`, `uint8` 0–255 or float 0–1).

### Multi-modal inputs

Add depth maps, camera intrinsics, or camera extrinsics from your own data loader:

```python
inputs = {
    "images": image_paths,
    "depth": depth_maps,     # optional
    "intrinsic": K,          # optional
    "extrinsic": poses,      # optional
}
predictions = model.predict(inputs)
```

For **S images**, use NumPy arrays or PyTorch tensors **without a batch dimension**. `H₀, W₀` are the original image height and width:

| Field | Shape | Meaning |
| --- | --- | --- |
| `depth` | `[S, H₀, W₀]` or a list of 2D arrays | Camera-z depth in meters; zero means unobserved |
| `intrinsic` | `[3, 3]` or `[S, 3, 3]` | Shared or per-image pinhole camera intrinsics |
| `extrinsic` | `[S, 3, 4]` or `[S, 4, 4]` | World-to-camera extrinsics; translation in meters |

- Supply geometry for the **original images**, in the same frame order. Resizing is automatic.
- Omit unavailable fields or set them to `None`. Use **zero arrays** for missing depth frames; provided camera parameters must be valid for every frame.
- Camera axes: x right, y down, z forward. Sparse depth is supported; resizing/cropping can discard isolated samples.

### Inference modes

`group_size` (`G`) sets the number of frames processed together.

```python
predictions = model.predict(inputs, group_size=2, queue_capacity=50)
```

| Mode | Arguments | Behavior |
| --- | --- | --- |
| Offline | `group_size=None` (default) | Reconstruct all images jointly |
| Online | `group_size=1` | Process one image at a time |
| Grouped | `group_size=G` | Process groups of up to G images |
| Hybrid offline-online | `group_size=G, hybrid=True` | Process the first group offline, then continue online |

`queue_capacity` (`Q`) is the KV-cache queue capacity in frames, including the current group. It defaults to the input sequence length and must hold each complete group. Offline mode ignores it.

## Outputs

All geometric predictions are in metric scale, including with RGB-only input. Point coordinates, camera translations, and camera-z depth are in meters.

Tensor outputs are on **CPU**, without a batch dimension. `S` is the number of images; `H, W` are the **processed image height and width**.

| Key | Shape | Meaning |
| --- | --- | --- |
| `cam_points` | `[S, H, W, 3]` | Local point maps in each camera's coordinates |
| `world_points` | `[S, H, W, 3]` | Point maps in a common coordinate system |
| `extrinsics` | `[S, 3, 4]` | World-to-camera extrinsics in that common system |
| `cam_points_conf` | `[S, H, W]` | Per-point confidence, not a probability |
| `images` | `[S, 3, H, W]` | Processed RGB images in `[0, 1]` |
| `metadata` | dict | Frame order and image preprocessing transforms |

The first predicted camera defines the common coordinate system. Get camera-z depth with:

```python
depth = predictions["cam_points"][..., 2]  # [S, H, W]
```

## Visualization

Install the optional viewer:

```bash
python -m pip install -e '.[demo]'
```

```python
from unit.visualization import visualize

visualize(predictions)
```

Or run the RGB demo with PLY export:

```bash
python demo_viser.py --repo-id Haotian-sx/UniT \
  --images examples/HKUST_Toy/images --save-ply
```

Use `--checkpoint PATH` for local weights. Both loading options also work with `infer.py`, which runs without a viewer.

Open the printed URL. Exports are saved in a new folder under `./outputs/`. Run either script with `--help` for more options.

## TODO

- [x] Inference code
- [ ] VGGT-Long-style framework integration
- [ ] Training code
- [ ] Evaluation protocol

## Citation

BibTeX for the arXiv preprint:

```bibtex
@misc{wang2026unit,
  title={UniT: Unified Geometry Learning with Group Autoregressive Transformer},
  author={Haotian Wang and Yusong Huang and Zhaonian Kuang and Hongliang Lu and Xinhu Zheng and Meng Yang and Gang Hua},
  year={2026},
  eprint={2605.21131},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2605.21131}
}
```

## Acknowledgements & License

We thank [VGGT](https://github.com/facebookresearch/vggt), [CUT3R](https://github.com/CUT3R/CUT3R), [Pi3](https://github.com/yyfz/Pi3), and [StreamVGGT](https://github.com/wzzheng/StreamVGGT) for their open-source contributions. Visualization uses [Viser](https://github.com/nerfstudio-project/viser) and optional [Sky Segmentation](https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing).

See [LICENSE.txt](LICENSE.txt) and [Third-party notices](THIRD_PARTY_NOTICES.md) for the code. The pretrained weights are distributed under [CC BY-NC 4.0](https://huggingface.co/Haotian-sx/UniT/blob/main/LICENSE.txt).
