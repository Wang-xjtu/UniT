"""Group causal masks and the fixed-capacity queue used during inference."""


import torch
import torch.nn as nn


class QueueKVCache(nn.Module):
    """Store per-frame keys and values in a bounded logical queue.

    The released model uses stride-based dropping: a rotating candidate selects
    an old frame whenever a new frame needs space. All frames from the current
    input group remain visible together, preserving bidirectional attention
    within that group. Returned frames follow their original sequence order.

    Queue capacity is measured in complete frames. Cache buffers are temporary
    and are not part of the model checkpoint.
    """

    def __init__(
        self,
        batch_size: int,
        tokens_per_frame: int,
        queue_capacity: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        for name, value in (
            ("batch_size", batch_size), ("tokens_per_frame", tokens_per_frame),
            ("queue_capacity", queue_capacity), ("num_kv_heads", num_kv_heads),
            ("head_dim", head_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError("Cache dtype must be a floating-point torch dtype")

        self.tokens_per_frame = tokens_per_frame
        self.queue_capacity = queue_capacity
        self.step = 1

        # Physical storage: [batch, heads, capacity * tokens_per_frame, head_dim].
        cache_shape = (
            batch_size,
            num_kv_heads,
            queue_capacity * tokens_per_frame,
            head_dim,
        )

        self.register_buffer(
            "k_cache",
            torch.zeros(cache_shape, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(cache_shape, dtype=dtype, device=device),
            persistent=False,
        )

        # Token offsets within one frame, reused for gather/scatter operations.
        self.register_buffer(
            "frame_offsets",
            torch.arange(tokens_per_frame, device=device),
            persistent=False,
        )

        # Logical ordering is maintained on CPU; each slot is a physical token offset.
        self.all_phys_starts = [i * tokens_per_frame for i in range(queue_capacity)]
        self.active_slots: list[int] = []
        self.free_slots: list[int] = self.all_phys_starts.copy()
        # Fast path: before the cache ever rolls over, active slots stay as a
        # contiguous prefix and we can return cheap views instead of gather copies.
        self.append_only = True
        self.used_frames = 0

    @torch.no_grad()
    def update(
        self, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append one input group and return retained keys and values.

        Both tensors must have shape [B, H, G * P, D], match the cache device
        and dtype, and contain complete frames. ``P`` is tokens_per_frame and
        ``G`` must fit the queue. Validation precedes every state mutation.
        """
        num_frames_in = self._validate_group(k_new, v_new)
        seq_len = k_new.shape[2]

        if num_frames_in == 0:
            return self._get_logical_cache()

        # Fast path for the common online case: while the cache has not rolled
        # over yet, frames are appended sequentially and can be read as a prefix.
        if self.append_only and len(self.free_slots) >= num_frames_in:
            start_frame = self.used_frames
            start_token = start_frame * self.tokens_per_frame
            end_token = start_token + seq_len

            self.k_cache[:, :, start_token:end_token, :].copy_(k_new)
            self.v_cache[:, :, start_token:end_token, :].copy_(v_new)

            new_slots = self.all_phys_starts[start_frame : start_frame + num_frames_in]
            self.active_slots.extend(new_slots)
            del self.free_slots[:num_frames_in]
            self.used_frames += num_frames_in

            return self._get_logical_cache()

        # Once we need to evict/roll over, fall back to the general indexed path.
        self.append_only = False

        # Reserve distinct slots for the full incoming group before device writes.
        write_target_slots = []
        current_group_slots = set()
        for _ in range(num_frames_in):
            if not self.free_slots:
                pop_idx = self.step % self.queue_capacity
                self.step += 1
                # Skip this group's newly reserved slots without another stride
                # increment. Only previously cached history can be discarded.
                while self.active_slots[pop_idx] in current_group_slots:
                    pop_idx = (pop_idx + 1) % len(self.active_slots)
                released = self.active_slots.pop(pop_idx)
                self.free_slots.append(released)

            target_slot = self.free_slots.pop(0)
            write_target_slots.append(target_slot)
            self.active_slots.append(target_slot)
            current_group_slots.add(target_slot)

        write_starts = torch.tensor(
            write_target_slots, device=self.k_cache.device
        ).unsqueeze(1)
        write_indices = (write_starts + self.frame_offsets).view(-1)
        self.k_cache.index_copy_(2, write_indices, k_new)
        self.v_cache.index_copy_(2, write_indices, v_new)

        # Expose retained frames in their logical sequence order.
        return self._get_logical_cache()

    def _validate_group(self, keys: torch.Tensor, values: torch.Tensor) -> int:
        """Reject malformed input before changing slot lists or cache buffers."""
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise ValueError("Keys and values must be torch tensors")
        if keys.ndim != 4 or values.ndim != 4 or keys.shape != values.shape:
            raise ValueError("Keys and values must have identical [B, H, G * P, D] shapes")
        expected = (self.k_cache.shape[0], self.k_cache.shape[1], self.k_cache.shape[3])
        if (keys.shape[0], keys.shape[1], keys.shape[3]) != expected:
            raise ValueError("Key/value batch, head count, and head dimension must match the cache")
        if keys.layout != torch.strided or values.layout != torch.strided:
            raise ValueError("Keys and values must use a dense strided tensor layout")
        if keys.device != self.k_cache.device or values.device != self.v_cache.device:
            raise ValueError("Keys and values must be on the same device as the cache")
        if keys.dtype != self.k_cache.dtype or values.dtype != self.v_cache.dtype:
            raise ValueError("Keys and values must have the same dtype as the cache")
        tokens = keys.shape[2]
        if tokens % self.tokens_per_frame:
            raise ValueError("Key/value token count must contain complete frames")
        frames = tokens // self.tokens_per_frame
        if frames > self.queue_capacity:
            raise ValueError("The complete input group must fit within the queue capacity")
        return frames

    def _get_logical_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return retained frames in sequence order, using a view before rollover."""
        if self.append_only:
            used_tokens = self.used_frames * self.tokens_per_frame
            return (
                self.k_cache[:, :, :used_tokens, :],
                self.v_cache[:, :, :used_tokens, :],
            )

        if not self.active_slots:
            # Return correctly shaped empty views.
            return (self.k_cache[:, :, :0, :], self.v_cache[:, :, :0, :])

        # Expand logical frame slots into token indices.
        read_starts = torch.tensor(
            self.active_slots, device=self.k_cache.device, dtype=torch.long
        ).unsqueeze(1)

        read_indices = (read_starts + self.frame_offsets).view(-1)

        return (
            torch.index_select(self.k_cache, 2, read_indices),
            torch.index_select(self.v_cache, 2, read_indices),
        )


@torch.no_grad()
def get_group_causal_mask(
    P: int,
    S: int,
    group_size: int | torch.Tensor,
    device: torch.device | str | None = None,
    *,
    batch_size: int | None = None,
) -> torch.Tensor:
    """Build an SDPA boolean mask with bidirectional groups and causal history.

    Args:
        P: Number of tokens per frame; use one for the camera head.
        S: Number of frames in the sequence.
        group_size: Positive integer shared by the batch, or integer Tensor [B]
            containing each sequence's group size in frames.
        device: Output device; defaults to the tensor's device, or CPU for an int.
        batch_size: Expected batch size. With an int, defaults to one; with a
            tensor, defaults to its length. Model callers pass their actual B.

    Returns:
        Boolean tensor [B, 1, S * P, S * P]. True allows a query to attend to a
        key in the same or an earlier group. Evaluation applies this ordering
        through grouped scheduling and retained KV history instead of this mask.
    """
    if batch_size is not None and (
        isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")

    if isinstance(group_size, int) and not isinstance(group_size, bool):
        if group_size < 1:
            raise ValueError("group_size values must be positive")
        B = 1 if batch_size is None else batch_size
        group_sizes = torch.full((B,), group_size, dtype=torch.long, device=device)
    elif isinstance(group_size, torch.Tensor):
        integer_dtypes = (
            torch.uint8, torch.uint16, torch.uint32, torch.uint64,
            torch.int8, torch.int16, torch.int32, torch.int64,
        )
        if group_size.dtype not in integer_dtypes:
            raise ValueError("group_size must be an integer or an integer Tensor [B]")
        if group_size.ndim != 1 or group_size.numel() == 0:
            raise ValueError("group_size must be a non-empty Tensor with shape [B]")
        B = group_size.shape[0]
        if batch_size is not None and B != batch_size:
            raise ValueError("group_size must contain one value per batch item")
        group_sizes = group_size.to(device=device, dtype=torch.long)
        if torch.any(group_sizes <= 0):
            raise ValueError("group_size values must be positive")
    else:
        raise ValueError("group_size must be an integer or an integer Tensor [B]")

    L = S * P
    token_ids = torch.arange(L, device=group_sizes.device).unsqueeze(0).expand(B, L)
    tokens_per_group = (P * group_sizes).view(B, 1)
    group_ids = token_ids // tokens_per_group
    allowed = group_ids.unsqueeze(2) >= group_ids.unsqueeze(1)
    return allowed.unsqueeze(1)
