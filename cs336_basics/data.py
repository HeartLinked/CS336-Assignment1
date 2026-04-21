import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of language modeling sequences from a dataset.

    Supports both in-memory numpy arrays and memory-mapped arrays (np.memmap
    or arrays loaded with mmap_mode='r'), so the full dataset does not need
    to fit in RAM.

    Args:
        dataset: 1-D numpy array (or memmap) of integer token IDs.
        batch_size: Number of sequences to sample.
        context_length: Length of each sequence.
        device: PyTorch device string (e.g. 'cpu' or 'cuda:0').

    Returns:
        (x, y): two LongTensors of shape (batch_size, context_length).
        y[i] == x[i] shifted left by one (next-token targets).
    """
    max_start = len(dataset) - context_length
    start_indices = torch.randint(0, max_start, (batch_size,))

    # Explicitly copy slices into contiguous numpy arrays before converting to
    # tensors.  This is necessary for memmap arrays, which are not contiguous
    # in memory and cannot be wrapped by torch directly.
    x = torch.stack([
        torch.from_numpy(np.array(dataset[i: i + context_length], dtype=np.int64))
        for i in start_indices
    ])
    y = torch.stack([
        torch.from_numpy(np.array(dataset[i + 1: i + context_length + 1], dtype=np.int64))
        for i in start_indices
    ])

    return x.to(device), y.to(device)
