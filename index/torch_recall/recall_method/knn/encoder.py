from __future__ import annotations

import torch


def encode_query(
    query_vector,
    meta: dict,
) -> torch.Tensor:
    """Convert a query vector into the [1, D] tensor expected by KNNRecall.forward().

    Args:
        query_vector: list, numpy ndarray, or torch.Tensor of shape [D] or [1, D].
        meta: the meta dict returned by KNNBuilder.build().
    Returns:
        [1, D] float tensor.
    """
    if not isinstance(query_vector, torch.Tensor):
        query_vector = torch.tensor(query_vector, dtype=torch.float32)
    else:
        query_vector = query_vector.float()

    if query_vector.ndim == 1:
        query_vector = query_vector.unsqueeze(0)  # [D] -> [1, D]

    D = meta["embedding_dim"]
    if query_vector.shape[-1] != D:
        raise ValueError(
            f"Query dimension {query_vector.shape[-1]} != embedding_dim {D}"
        )

    return query_vector


def save_query_tensors(
    query_vector,
    meta: dict,
    path: str,
) -> None:
    """Encode a query vector and save for C++ inference.

    File format: list[list[Tensor]] with a single batch containing
    one tensor (the query).
    """
    q = encode_query(query_vector, meta)
    torch.save([[q]], path)
