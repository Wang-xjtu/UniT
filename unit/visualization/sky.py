"""Optional sky filtering using the exact transforms applied to model images."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from unit.inference.preprocessing import transform_mask
from unit.utils.runtime import resolve_sky_model_path


def segment_sky(
    image_path: str | Path, onnx_session, mask_filename: str | Path | None = None,
) -> np.ndarray:
    """Return 255 for non-sky pixels using the pretrained sky filter.

    The model receives ImageNet-normalized RGB. Its first output is min-max
    normalized to uint8, resized to the source image, and thresholded below 32.
    A constant prediction contains no usable separation, so all pixels are kept.
    """
    import cv2

    # Match Pillow RGB preprocessing: geometry refers to stored pixel coordinates.
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    h, w = image.shape[:2]

    inputs = onnx_session.get_inputs()
    if len(inputs) != 1:
        raise ValueError("The sky segmentation model must have one image input")
    input_spec = inputs[0]
    if len(input_spec.shape) != 4 or input_spec.type != "tensor(float)":
        raise ValueError("The sky segmentation model must accept an NCHW float32 image")
    batch_size, channels, model_height, model_width = input_spec.shape
    if isinstance(batch_size, int) and batch_size != 1:
        raise ValueError("The sky segmentation model must accept batch size 1")
    if isinstance(channels, int) and channels != 3:
        raise ValueError("The sky segmentation model must accept three RGB channels")
    # Use the established sky-filter resolution for dynamic spatial axes.
    input_height = model_height if isinstance(model_height, int) and model_height > 0 else 320
    input_width = model_width if isinstance(model_width, int) and model_width > 0 else 320
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (input_width, input_height)).astype(np.float32)
    image_normalized = (image_resized / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image_input = np.transpose(image_normalized, (2, 0, 1))[None].astype(np.float32)

    result = onnx_session.run(None, {input_spec.name: image_input})[0][0, 0]
    if not np.isfinite(result).all():
        raise ValueError("Sky segmentation returned non-finite values")
    minimum, maximum = np.min(result), np.max(result)
    if maximum == minimum:
        mask = np.full((h, w), 255, dtype=np.uint8)
    else:
        result = ((result - minimum) / (maximum - minimum) * 255).astype(np.uint8)
        result = cv2.resize(result, (w, h))
        mask = (result < 32).astype(np.uint8) * 255

    if mask_filename is not None:
        path = Path(mask_filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        success, encoded = cv2.imencode(path.suffix, mask)
        if not success:
            raise ValueError(f"Could not encode sky mask: {path}")
        with path.open("xb") as stream:
            stream.write(encoded.tobytes())

    return mask



def apply_sky_segmentation(
    conf: np.ndarray,
    image_paths: Sequence[str | Path],
    sky_masks_dir: str | Path,
    skyseg_path: str | Path | None = None,
    *,
    transforms: Sequence[dict],
) -> np.ndarray:
    """Filter a copy of confidence maps and write original-resolution masks separately."""
    if len(image_paths) != len(conf) or len(transforms) != len(conf):
        raise ValueError("Sky filtering requires one image and transform per confidence map")
    model_path = resolve_sky_model_path(skyseg_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Sky segmentation model not found: {model_path}. "
            "Supply --skyseg-path or disable --mask-sky."
        )
    try:
        import onnxruntime
    except ImportError as exc:
        raise ImportError("Sky filtering requires: pip install -e '.[demo,sky]'") from exc
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    destination = Path(sky_masks_dir)
    destination.mkdir(parents=True, exist_ok=False)
    masks = []
    for index, (image_path, transform) in enumerate(zip(image_paths, transforms)):
        mask_path = destination / f"{index:06d}_{Path(image_path).stem}.png"
        mask = segment_sky(image_path, session, mask_path)
        mask = transform_mask(mask, transform)
        if mask.shape != conf.shape[1:]:
            raise ValueError("Sky transform and prediction resolution do not match")
        masks.append(mask > 0)
    return conf * np.stack(masks).astype(conf.dtype)
