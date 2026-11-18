"""Public loading and single-sequence prediction without changing model structure."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

import torch

from unit.inference._cache import release_caches
from unit.inference.engine import run_inference
from unit.inference.preprocessing import prepare_inputs
from unit.inference.scheduling import make_schedule
from unit.utils.checkpoint_compat import load_model_state_dict
from unit.utils.runtime import inference_precision

Model = TypeVar("Model", bound=torch.nn.Module)


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Choose CUDA when available, otherwise CPU; accept explicit CUDA indices."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    selected = torch.device(device)
    if selected.type not in {"cuda", "cpu"}:
        raise ValueError("device must be 'auto', 'cpu', or a CUDA device")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return selected


def from_checkpoint(
    model_class: type[Model],
    checkpoint: str | Path,
    *,
    device: str | torch.device = "auto",
) -> Model:
    """Construct the released architecture and strictly load a local checkpoint."""
    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    selected = resolve_device(device)
    model = model_class()
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_model

        load_model(model, path, strict=True, device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
        load_model_state_dict(model, state, strict=True)
        del state
    return model.to(selected).eval()


def _load_hub_model(
    model_class: type[Model],
    repo_id: str,
    *,
    filename: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    token: str | bool | None = None,
    device: str | torch.device = "auto",
    force_download: bool = False,
) -> Model:
    """Download weights from the Hub and reuse the strict file loader."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.constants import HF_HUB_OFFLINE, SAFETENSORS_SINGLE_FILE
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    download = partial(
        hf_hub_download, repo_id=repo_id, repo_type="model",
        revision=revision, cache_dir=cache_dir,
        local_files_only=local_files_only, token=token, force_download=force_download,
    )
    if filename is not None:
        checkpoint = download(filename=filename)
    else:
        try:
            checkpoint = download(filename=SAFETENSORS_SINGLE_FILE)
        except LocalEntryNotFoundError:
            if not (local_files_only or HF_HUB_OFFLINE):
                raise
            checkpoint = download(filename="checkpoint.pt")
        except EntryNotFoundError:
            checkpoint = download(filename="checkpoint.pt")
    return from_checkpoint(model_class, checkpoint, device=device)


def _unpack_inputs(inputs: Any) -> tuple[Any, Any, Any, Any]:
    """Read one raw sequence dict, or accept images alone as RGB shorthand."""
    if not isinstance(inputs, Mapping):
        return inputs, None, None, None
    unknown = set(inputs) - {"images", "depth", "intrinsic", "extrinsic"}
    if unknown:
        names = ", ".join(sorted(repr(name) for name in unknown))
        raise ValueError(
            f"Unknown input fields: {names}. "
            "Allowed fields are images, depth, intrinsic, and extrinsic"
        )
    if "images" not in inputs or inputs["images"] is None:
        raise ValueError("Input dictionary requires a non-None 'images' field")
    return inputs["images"], inputs.get("depth"), inputs.get("intrinsic"), inputs.get("extrinsic")


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    inputs: Any,
    *,
    group_size: int | None = None,
    queue_capacity: int | None = None,
    hybrid: bool = False,
    target_size: int = 518,
    preprocess_mode: str = "crop",
    postprocess: bool = False,
) -> dict[str, Any]:
    """Predict one sequence in processed image coordinates and return CPU tensors.

    Pass a dictionary containing ``images`` and optional ``depth``, ``intrinsic``,
    and ``extrinsic`` fields. Images can be a directory, ordered paths, or RGB
    NumPy/PIL images. Geometry uses original image coordinates, without a batch
    axis. Images alone are accepted as an RGB shorthand. ``hybrid=True`` uses
    hybrid offline-online inference: one joint group, then one frame at a time.

    See the README for input/output shapes and coordinates. No files are written.
    Model training flags are restored after the call; caches are not shared
    between calls. Prepared inputs and completed outputs stay on CPU; only the
    current group is transferred to the model device. ``queue_capacity=None``
    retains up to the input sequence's frame count in cached modes.
    """
    if not isinstance(postprocess, bool):
        raise ValueError("postprocess must be a boolean")
    release_caches(model)
    modes = [(module, module.training) for module in model.modules()]
    try:
        images, depths, intrinsics, extrinsics = _unpack_inputs(inputs)
        batch, metadata = prepare_inputs(
            images, depths=depths, intrinsics=intrinsics, extrinsics=extrinsics,
            target_size=target_size, preprocess_mode=preprocess_mode,
        )
        schedule = make_schedule(
            batch["images"].shape[1], group_size, queue_capacity, hybrid
        )
        parameter = next(model.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        processed_images = batch["images"][0]
        precision = inference_precision(device)
        model.eval()
        with torch.autocast(device_type=device.type, dtype=precision,
                            enabled=device.type == "cuda"):
            output = run_inference(
                batch, model, schedule, postprocess=postprocess,
            )
        result = {key: value[0].detach().cpu() for key, value in output.items()}
        result["images"] = processed_images
        result["metadata"] = metadata
        return result
    finally:
        release_caches(model)
        for module, training in modes:
            module.training = training
