<div align="center">

# UniT

**Unified Geometry Learning with Group Causal Transformer**

Haotian Wang · Yusong Huang · Zhaonian Kuang · Hongliang Lu<br>
Xinhu Zheng · Meng Yang · Gang Hua

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2605.21131)
[![Project Page](https://img.shields.io/badge/Project-Page-2563eb)](https://omega-hkustgz.github.io/projects/unit/)
[![Hugging Face Demo](https://img.shields.io/badge/Hugging%20Face-Demo-ffcc4d)](https://enceladush-unit.hf.space/)

[Usage](#usage) · [Input](#input) · [Output](#output) · [Visualization](#visualization)

</div>

**UniT** is a feed-forward model for unified geometry learning.

- **Any view configuration:** online, multi-view streaming, and offline inference from one or more views.
- **Flexible modality combinations:** RGB images with optional depth maps and camera parameters.
- **Metric-scale perception:** 3D geometry predicted in meters.
- **Long-horizon scalability:** long-sequence inference with bounded memory cost.

## News

- **2026-11-18:** Inference code and pretrained weights released.
- **2026-05-21:** Paper, project page, and Hugging Face demo released.

<a name="quick-start"></a>

## Usage

Reconstruct the bundled image sequence from the repository root.

```python
from unit import UniT

model = UniT.from_pretrained("Haotian-sx/UniT")
inputs = {"images": "examples/HKUST_Toy/images"}
predictions = model.predict(inputs)
print(predictions["world_points"].shape)
```

With a [local checkpoint](https://huggingface.co/Haotian-sx/UniT/resolve/main/checkpoint.pt), use `model = UniT.from_checkpoint("checkpoints/checkpoint.pt")`.

### Installation

Use Python 3.11.

```bash
git clone https://github.com/Wang-xjtu/UniT.git
cd UniT
python -m pip install -e .
```

### Inference modes

| Mode | Arguments | Behavior |
| --- | --- | --- |
| Offline | `group_size=None` (default) | Process all frames together |
| Online | `group_size=1` | Process one frame at a time |
| Grouped | `group_size=G` | Process up to G frames at a time |
| Hybrid | `group_size=G, hybrid=True` | Process up to G frames first, then one at a time |

In cached modes, `queue_capacity` defaults to the number of input frames. Set a smaller value to limit the cache, with enough room for a full group.

## Input

### RGB only

Pass an image directory or an ordered list of paths, PIL images, or RGB NumPy arrays as `images`. Files in a directory are read in filename order. RGB arrays have shape `[H, W, 3]` with `uint8` values in 0–255 or floats in 0–1.

### Multi-modal

Add any geometry you have to the same `inputs` dictionary before calling `predict()`.

```python
inputs.update(depth=depth_maps, intrinsic=K, extrinsic=poses)
```

Use NumPy arrays or PyTorch tensors without a batch dimension. Match the original images and their order. `S` is the number of frames, and `H₀` and `W₀` are the original image dimensions.

| Optional field | Shape | Meaning |
| --- | --- | --- |
| `depth` | `[S, H₀, W₀]` or a list of 2D maps | Camera-z depth in meters |
| `intrinsic` | `[3, 3]` or `[S, 3, 3]` | Shared or per-frame intrinsics |
| `extrinsic` | `[S, 3, 4]` or `[S, 4, 4]` | World-to-camera transforms with translation in meters |

Omit fields you do not have. Use zeros for missing depth pixels or frames. Camera parameters, when provided, must cover every frame. Camera axes are x right, y down, and z forward.

## Output

Point coordinates, camera translations, and depth are in meters, even with RGB-only input. `H` and `W` are the processed image dimensions. To get camera-z depth:

```python
depth = predictions["cam_points"][..., 2]  # [S, H, W]
```

| Key | Shape | Meaning |
| --- | --- | --- |
| `cam_points` | `[S, H, W, 3]` | Point maps in each frame's camera coordinates |
| `world_points` | `[S, H, W, 3]` | Point maps in the first predicted camera's coordinates |
| `extrinsics` | `[S, 3, 4]` | World-to-camera transforms |
| `cam_points_conf` | `[S, H, W]` | Per-point confidence |

## Visualization

```bash
python -m pip install -e '.[demo]'
python demo_viser.py
```

Open the URL printed in your terminal. The demo writes `scenes.ply` to a new folder under `outputs/`. To use your own images, add `--images /path/to/images`.

- `--no-save-ply` opens the viewer without exporting a point cloud.
- `--no-viewer` exports the point cloud and exits.

To view predictions you already have in Python:

```python
from unit.visualization import visualize

visualize(predictions)
```

Pass `save_ply=True` to export a point cloud from Python. Both the viewer and PLY use an **80 m camera-z depth limit** by default. Set `--max-depth 0` on the command line or `max_depth=0` in Python to disable the limit. `--conf-threshold` (`init_conf_threshold` in Python) is the percentage of low-confidence points to filter out. For other options, run `python demo_viser.py --help`.

---

<details>
<summary>TODO, citation and acknowledgements</summary>

### TODO

- [x] Inference code
- [ ] VGGT-Long-style framework integration
- [ ] Training code
- [ ] Evaluation protocol

### Citation

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

### Acknowledgements

We thank [VGGT](https://github.com/facebookresearch/vggt), [CUT3R](https://github.com/CUT3R/CUT3R), [Pi3](https://github.com/yyfz/Pi3), and [StreamVGGT](https://github.com/wzzheng/StreamVGGT) for their open-source contributions. Visualization uses [Viser](https://github.com/nerfstudio-project/viser) and optional [Sky Segmentation](https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing).

</details>

See [LICENSE.txt](LICENSE.txt) and [Third-party notices](THIRD_PARTY_NOTICES.md) for the code. The pretrained weights are distributed under [CC BY-NC 4.0](https://huggingface.co/Haotian-sx/UniT/blob/main/LICENSE.txt).
