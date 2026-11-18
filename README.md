<div align="center">

# UniT

**Unified Geometry Learning with Group Causal Transformer**

Haotian Wang · Yusong Huang · Zhaonian Kuang · Hongliang Lu<br>
Xinhu Zheng · Meng Yang · Gang Hua

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2605.21131)
[![Project Page](https://img.shields.io/badge/Project-Page-2563eb)](https://omega-hkustgz.github.io/projects/unit/)
[![Hugging Face Demo](https://img.shields.io/badge/Hugging%20Face-Demo-ffcc4d)](https://enceladush-unit.hf.space/)

[Quick Start](#quick-start) · [Python API](#python-api) · [Inputs & Modes](#optional-inputs-and-modes) · [Visualization](#visualization)

</div>

**UniT** is a feed-forward model for unified geometry learning.

- **Any view configuration:** online, multi-view streaming, and offline inference from one or more views.
- **Flexible modality combinations:** RGB images with optional depth maps and camera parameters.
- **Metric-scale perception:** 3D geometry predicted in meters.
- **Long-horizon scalability:** long-sequence inference with bounded memory cost.

## News

- **2026-11-18:** Inference code and pretrained weights released.
- **2026-05-21:** Paper, project page, and Hugging Face demo released.

<a name="installation"></a>

## Quick Start

Use **Python 3.11** to install UniT and run the included HKUST_Toy example:

```bash
git clone https://github.com/Wang-xjtu/UniT.git
cd UniT
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[demo]'
python demo_viser.py
```

[Weights](https://huggingface.co/Haotian-sx/UniT) download automatically on first use. The demo opens the viewer and exports `scenes.ply` to a new folder under `./outputs/`. Open the printed URL to explore the scene.

For your own images, add `--images /path/to/images`. Use `--checkpoint PATH` for local weights and `--output-dir PATH` to choose a new export directory.

- `--no-viewer`: export the PLY and exit.
- `--no-save-ply`: open the viewer without PLY export. Optional sky filtering still saves masks.

Run `python demo_viser.py --help` for all options.

## Python API

```python
from unit import UniT

model = UniT.from_pretrained("Haotian-sx/UniT")
inputs = {"images": "examples/HKUST_Toy/images"}
predictions = model.predict(inputs)
depth = predictions["cam_points"][..., 2]  # [S, H, W], meters
```

For [downloaded weights](https://huggingface.co/Haotian-sx/UniT/resolve/main/checkpoint.pt), replace the loading line with `model = UniT.from_checkpoint("checkpoints/checkpoint.pt")`.

`images` accepts a directory (filename order), an ordered list of paths, or a list of PIL images or RGB NumPy arrays (`[H, W, 3]`, `uint8` 0–255 or float 0–1). CUDA is used when available, otherwise CPU. `predict()` returns results in memory without writing files.

### Outputs

UniT predicts metric-scale geometry, including from RGB alone. **Point coordinates, camera translations, and camera-z depth are in meters.** Tensor outputs are on CPU without a batch dimension; `S` is the image count and `H, W` are the processed image dimensions.

| Key | Shape | Meaning |
| --- | --- | --- |
| `cam_points` | `[S, H, W, 3]` | Local point maps in each camera's coordinates |
| `world_points` | `[S, H, W, 3]` | Point maps in a common coordinate system |
| `extrinsics` | `[S, 3, 4]` | World-to-camera extrinsics in that common system |
| `cam_points_conf` | `[S, H, W]` | Per-point confidence, not a probability |
| `images` | `[S, 3, H, W]` | Processed RGB images in `[0, 1]` |
| `metadata` | dict | Frame order and image preprocessing transforms |

The first predicted camera defines the common coordinate system.

## Optional Inputs and Modes

### Geometric inputs

Add any available geometry from your data loader:

```python
inputs = {
    "images": image_paths,
    "depth": depth_maps,
    "intrinsic": K,
    "extrinsic": poses,
}
predictions = model.predict(inputs)
```

Geometry uses NumPy arrays or PyTorch tensors **without a batch dimension**, matching the **original images** and their frame order. `H₀, W₀` are the original height and width; resizing is automatic.

| Optional field | Shape | Meaning |
| --- | --- | --- |
| `depth` | `[S, H₀, W₀]` or a list of 2D arrays | Camera-z depth in meters; zero means unobserved |
| `intrinsic` | `[3, 3]` or `[S, 3, 3]` | Shared or per-image pinhole camera intrinsics |
| `extrinsic` | `[S, 3, 4]` or `[S, 4, 4]` | World-to-camera extrinsics; translation in meters |

Omit unavailable fields or set them to `None`. Use zero arrays for missing depth frames; provided camera parameters must be valid for every frame. Camera axes are x right, y down, z forward. Sparse depth is supported, though resizing/cropping can discard isolated samples.

### Inference modes

```python
predictions = model.predict(inputs, group_size=2, queue_capacity=50)
```

| Mode | Arguments | Behavior |
| --- | --- | --- |
| Offline | `group_size=None` (default) | Reconstruct all images jointly |
| Online | `group_size=1` | Process one image at a time |
| Grouped | `group_size=G` | Process groups of up to G images |
| Hybrid offline-online | `group_size=G, hybrid=True` | Process the first group offline, then continue online |

`queue_capacity` (`Q`) limits cached frames, including the current group. It defaults to the sequence length and must fit each group; offline mode ignores it. Set a fixed Q for long sequences. CLI equivalents are `--group-size`, `--queue-capacity`, and `--hybrid`; geometric inputs use the Python API.

## Visualization

View a prediction dict from Python, optionally exporting a PLY:

```python
from unit.visualization import visualize

visualize(predictions, save_ply=True)
```

Omit `save_ply=True` to view without PLY export. Open the printed URL, drag to orbit, scroll to zoom, or click a camera to visit its view. Use the frame selector and playback controls to explore the sequence; stop the viewer with Ctrl+C.

Viewer and PLY use an **80 m camera-z depth limit**. Set `max_depth=0` in Python or `--max-depth 0` in the CLI to disable it. The confidence filter (`init_conf_threshold` / `--conf-threshold`) excludes the lowest-confidence percentage of points. These filters leave predictions unchanged; PLY uses the initial settings, while viewer controls affect only the display.

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
