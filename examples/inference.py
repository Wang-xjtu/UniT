"""A complete RGB prediction example; add available geometry to the input dict.

Run from the repository root with ``python examples/inference.py``. To use
observations from your data loader, add ``depth``, ``intrinsic``, and/or
``extrinsic`` to ``inputs`` in the same frame order and original image coordinates.
"""

from pathlib import Path

from unit import UniT


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    model = UniT.from_pretrained("Haotian-sx/UniT")
    # For local weights: model = UniT.from_checkpoint("checkpoints/checkpoint.pt")
    inputs = {"images": project_root / "examples/HKUST_Toy/images"}
    predictions = model.predict(inputs)
    print("Common-coordinate point maps:", tuple(predictions["world_points"].shape))
    print("World-to-camera poses:", tuple(predictions["extrinsics"].shape))

    # To inspect the result after installing the demo extra:
    # from unit.visualization import visualize
    # visualize(predictions)


if __name__ == "__main__":
    main()
