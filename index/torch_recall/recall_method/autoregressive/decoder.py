from __future__ import annotations

import torch
import torch.nn as nn


class MockDecoder(nn.Module):
    """Random-projection decoder for testing generative recall.

    Produces logits based only on user_repr (ignores partial_paths).
    """

    def __init__(self, user_dim: int, vocab_size: int = 2048):
        super().__init__()
        self.proj = nn.Linear(user_dim, vocab_size)
        self.vocab_size = vocab_size
        self.user_dim = user_dim

    def forward(
        self, user_repr: torch.Tensor, partial_paths: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            user_repr:     [B_beam, D] user representation.
            partial_paths: [B_beam, path_length] current partial sid paths (ignored).
        Returns:
            [B_beam, vocab_size] logits over next sid.
        """
        return self.proj(user_repr)
