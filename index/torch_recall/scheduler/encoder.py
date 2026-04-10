from __future__ import annotations

import torch

from torch_recall.recall_method.targeting.encoder import encode_user
from torch_recall.recall_method.knn.encoder import encode_query


def encode_pipeline_inputs(
    user_attrs: dict | None,
    query_vectors: list | None,
    meta: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode user attributes and query vectors into pipeline input tensors.

    Args:
        user_attrs: dict of user attributes for targeting, or None if
            no targeting leaf is present.
        query_vectors: list of query vectors (one per KNN leaf, in
            tree-walk order), or None if no KNN leaf is present.
            Each element can be a list, ndarray, or [D]/[1,D] tensor.
        meta: the pipeline meta dict returned by PipelineBuilder.build().

    Returns:
        (pred_satisfied [1, P] bool, query [1, D_total] float)
    """
    # --- targeting ---
    if "targeting" in meta and user_attrs is not None:
        pred_satisfied = encode_user(user_attrs, meta["targeting"])  # [P]
    else:
        pred_satisfied = torch.zeros(meta["num_preds"], dtype=torch.bool)
    pred_satisfied = pred_satisfied.unsqueeze(0)  # [1, P]

    # --- knn ---
    knn_leaves = meta.get("knn_leaves", [])
    if knn_leaves and query_vectors is not None:
        if len(query_vectors) != len(knn_leaves):
            raise ValueError(
                f"Expected {len(knn_leaves)} query vectors, got {len(query_vectors)}"
            )
        parts: list[torch.Tensor] = []
        for qv, leaf_meta in zip(query_vectors, knn_leaves):
            synthetic_meta = {"embedding_dim": leaf_meta["dim"]}
            parts.append(encode_query(qv, synthetic_meta))  # [1, D_i]
        query = torch.cat(parts, dim=1)  # [1, D_total]
    else:
        query = torch.zeros(1, meta["total_query_dim"])

    return pred_satisfied, query


def save_pipeline_tensors(
    user_attrs: dict | None,
    query_vectors: list | None,
    meta: dict,
    path: str,
) -> None:
    """Encode pipeline inputs and save for C++ inference."""
    pred_satisfied, query = encode_pipeline_inputs(
        user_attrs, query_vectors, meta
    )
    torch.save([[pred_satisfied, query]], path)
