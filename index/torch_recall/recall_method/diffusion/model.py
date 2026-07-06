"""Mock diffusion decoder for testing DiffusionRecall.

Unlike the autoregressive MockDecoder which outputs one vocabulary
distribution per step, this model outputs L distributions at once —
one per SID position — matching the parallel prediction semantics of
discrete diffusion (bidirectional attention over all positions).

A real decoder would condition on ``partial_paths`` (the partially
unmasked SID sequence) via a bidirectional Transformer.  This mock
ignores ``partial_paths`` and produces position logits purely from
the user representation, which is sufficient for structural tests.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MockDiffusionDecoder(nn.Module):
    """Random-projection bidirectional decoder for testing.

    Args:
        user_dim:    dimensionality of the input user representation.
        path_length: SID length L (number of positions).
        vocab_size:  vocabulary size V (default 2048).
    """

    MASK_ID: int = -1

    def __init__(
        self,
        user_dim: int,
        path_length: int,
        vocab_size: int = 2048,
    ):
        super().__init__()
        self.user_dim = user_dim
        self.path_length = path_length
        self.vocab_size = vocab_size
        self.proj = nn.Linear(user_dim, path_length * vocab_size)

    def forward(
        self,
        user_repr: torch.Tensor,
        partial_paths: torch.Tensor,
    ) -> torch.Tensor:
        """Produce all-position logits in a single forward pass.

        Args:
            user_repr:     ``[B_beam, user_dim]`` user representation.
            partial_paths: ``[B_beam, path_length]`` current partial SIDs;
                           unfilled positions carry ``MASK_ID`` (−1).
                           Ignored by this mock; a real model would
                           condition on them via bidirectional attention.
        Returns:
            ``[B_beam, path_length, vocab_size]`` logits.
        """
        B_beam = user_repr.shape[0]
        return self.proj(user_repr).view(B_beam, self.path_length, self.vocab_size)
