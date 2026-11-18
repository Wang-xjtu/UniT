"""Frame grouping independent of cache eviction and model computation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GroupSchedule:
    frames: int
    first_group: int
    queue_capacity: int
    hybrid: bool
    use_cache: bool

    def group_at(self, step: int) -> int:
        """Return the requested group size; the engine clips the final group."""
        return 1 if self.hybrid and step > 0 else self.first_group


def make_schedule(
    frames: int,
    group_size: int | None = None,
    queue_capacity: int | None = None,
    hybrid: bool = False,
) -> GroupSchedule:
    """Build a frame schedule with enough room for a complete cached group.

    Cache capacity defaults to the sequence length. In hybrid mode, omitting
    ``group_size`` uses that capacity for the first group, clipped to the
    available frames. Subsequent groups each contain one frame.
    """
    if queue_capacity is None:
        queue_capacity = frames
    for name, value in (("frames", frames), ("queue_capacity", queue_capacity)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if group_size is not None and (
        isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1
    ):
        raise ValueError("group_size must be None or a positive integer")
    if not isinstance(hybrid, bool):
        raise ValueError("hybrid must be a boolean")
    first = (queue_capacity if hybrid else frames) if group_size is None else group_size
    cached = hybrid or (group_size is not None and group_size != frames)
    if cached and min(first, frames) > queue_capacity:
        raise ValueError("queue_capacity must accommodate the complete input group")
    return GroupSchedule(frames, first, queue_capacity, hybrid, cached)
