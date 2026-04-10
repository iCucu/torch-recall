from __future__ import annotations

import torch
import torch.nn as nn


class RecallOp(nn.Module):
    """Base class for all recall operators.

    Every operator produces per-item scores with the unified signature::

        forward(pred_satisfied: [P] bool, query: [1, D] float) -> [1, N] float

    Positive scores indicate higher relevance; ``-inf`` means excluded.
    """
