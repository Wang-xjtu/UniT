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

Run RGB reconstruction on the HKUST_Toy example from the repository root.

```python
from unit import UniT

model = UniT.from_pretrained("Haotian-sx/UniT")
inputs = {"images": "examples/HKUST_Toy/images"}
predictions = model.predict(inputs)
print(predictions["world_points"].shape)
```

To load a [local checkpoint](https://huggingface.co/Haotian-sx/UniT/resolve/main/checkpoint.pt), use `model = UniT.from_checkpoint("checkpoints/checkpoint.pt")`.

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

```python
# For example, binocular streaming with an 8-frame cache queue 
predictions = model.predict(inputs, group_size=2, queue_capacity=8)
```

In cached modes, `queue_capacity` defaults to the number of input frames. Use a smaller value to cap the history, with enough room for a full group.

## Input

Only `images` is required. Add whichever geometric inputs you have to the same dictionary.

```python
inputs = {
    "images": image_paths,
    "depth": depth_maps,      # optional
    "intrinsic": intrinsics,  # optional
    "extrinsic": poses,       # optional
}
predictions = model.predict(inputs)
```

`S` is the number of frames, and `H₀` and `W₀` are the original image dimensions.

| Field | What to pass | Notes |
| --- | --- | --- |
| `images` | A directory, a list of paths, or a list of PIL images or RGB NumPy arrays | Directories are read in filename order. Lists keep their order. |
| `depth` | `[S, H₀, W₀]` or a list of 2D maps | Camera-z depth in meters. Use zeros for missing pixels and zero maps for missing frames. |
| `intrinsic` | `[3, 3]` or `[S, 3, 3]` | Shared or per-frame pinhole intrinsics in original image pixels |
| `extrinsic` | `[S, 3, 4]` or `[S, 4, 4]` | World-to-camera transforms with translation in meters |

Image directories support JPEG, PNG, BMP, TIFF, and WebP. RGB arrays have shape `[H₀, W₀, 3]`, with `uint8` values in 0–255 or floats in 0–1.

Provide geometry for the original images, in the same frame order, as NumPy arrays or PyTorch tensors without a batch dimension. UniT adjusts depth and intrinsics to match image resizing, cropping, and padding. Camera parameters, when provided, must be valid for every frame. Camera axes are x right, y down, and z forward.

## Output

The tensors in `predictions` are on CPU without a batch dimension. Point coordinates, camera translations, and depth are in meters, including when the input is RGB-only. Read camera-z depth from the local point map.

```python
depth = predictions["cam_points"][..., 2]  # [S, H, W]
```

| Key | Shape | Meaning |
| --- | --- | --- |
| `cam_points` | `[S, H, W, 3]` | Point maps in each frame's camera coordinates |
| `world_points` | `[S, H, W, 3]` | Point maps in the first predicted camera's coordinates |
| `extrinsics` | `[S, 3, 4]` | World-to-camera transforms |
| `cam_points_conf` | `[S, H, W]` | Per-point confidence |

`H` and `W` are the processed image dimensions. For the mapping from original to processed pixels, see `predictions["metadata"]["transforms"]`.

## Visualization

Install the demo extra, then pass `predictions` to the viewer.

```bash
python -m pip install -e '.[demo]'
```

```python
from unit.visualization import visualize

visualize(predictions)
```

Open the printed URL. Pass `save_ply=True` to also export a point cloud.

Run `python demo_viser.py` for HKUST_Toy. To try HKUST_INTR, start with every fourth frame.

```bash
python demo_viser.py --images examples/HKUST_INTR/images --stride 4
```

Drop `--stride 4` to use all 50 frames. To use your own sequence, pass `--images /path/to/images`. The demo saves `scenes.ply` in a new folder under `outputs/`.

- `--no-save-ply` opens the viewer without exporting a point cloud.
- `--no-viewer` exports the point cloud and exits.

The viewer and PLY use an **80 m camera-z depth limit** by default. Set `--max-depth 0` or `max_depth=0` in Python to disable it. The confidence filter hides points below the selected percentile. `--conf-threshold` defaults to 25, and `init_conf_threshold` in Python defaults to 50. PLY export uses the initial filters, and the prediction tensors keep all points.

Run `python demo_viser.py --help` for more options.

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
