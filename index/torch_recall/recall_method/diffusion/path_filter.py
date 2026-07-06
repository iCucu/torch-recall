"""SidPathFilter: path-bitmap constraint for discrete diffusion recall.

Each beam tracks a boolean mask over all N item paths. On every commit
the mask is narrowed to paths consistent with the chosen token. At decode
end the surviving set of paths directly identifies the candidate items —
no separate path-to-item lookup is required.

Contrast with the Trie constraint in autoregressive recall:

    AR (Trie):       node_valid [B, S] — propagated per request,
                     queried by current Trie state (O(1)/O(branch)).
    Diffusion (SPF): beam_mask [B, beam, N] — maintained per beam,
                     queried by position via scatter (O(N)).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SidPathFilter(nn.Module):
    """Path-level bitmap filter for non-sequential SID generation.

    Replaces the left-to-right Trie constraint used in autoregressive
    recall.  Each beam independently maintains a boolean mask over the
    N item paths; the mask shrinks after every token commit.

    The key correctness guarantee: after L commit steps, each beam's
    ``beam_mask`` contains exactly the item(s) whose SID matches the
    committed token sequence, so no separate path-to-item lookup is
    needed at resolve time.

    Args:
        path_vectors: ``[N, L]`` integer tensor — SID path for every item.
        vocab_size:   vocabulary size V (tokens in ``[0, V)``).
    """

    def __init__(self, path_vectors: torch.Tensor, vocab_size: int):
        super().__init__()
        self.register_buffer("path_vectors", path_vectors)  # [N, L]
        self.vocab_size = vocab_size
        self.num_paths = path_vectors.shape[0]
        self.path_length = path_vectors.shape[1]

    # ------------------------------------------------------------------

    def init_beam_mask(
        self, item_mask: torch.Tensor, beam_width: int
    ) -> torch.Tensor:
        """Broadcast per-user item mask to per-beam path mask.

        Args:
            item_mask:  ``[B, N]`` bool — targeting filter result.
            beam_width: number of beams.
        Returns:
            ``[B, beam, N]`` bool — identical across beams at init.
        """
        return item_mask.unsqueeze(1).expand(-1, beam_width, -1).clone()

    # ------------------------------------------------------------------

    def collect_valid_tokens(
        self, beam_mask: torch.Tensor, position: int
    ) -> torch.Tensor:
        """Return which tokens are valid at ``position`` for each beam.

        Token ``w`` is valid for beam ``(b, k)`` iff at least one active
        path has ``w`` at ``position``.

        Args:
            beam_mask: ``[B, beam, N]`` bool — current active-path mask.
            position:  depth index in ``[0, L)``.
        Returns:
            ``[B, beam, V]`` bool — True where the token is reachable.
        """
        B, bw, N = beam_mask.shape
        device = beam_mask.device
        tokens_at_pos = self.path_vectors[:, position]             # [N]

        # Expand token indices to match beam dimensions.
        # For inactive paths beam_mask.int() == 0, so scatter_add
        # contributes 0 — no false positives.
        idx = tokens_at_pos.view(1, 1, N).expand(B, bw, N)        # [B, beam, N]
        valid = torch.zeros(B, bw, self.vocab_size, dtype=torch.int32, device=device)
        valid.scatter_add_(2, idx.long(), beam_mask.int())
        return valid.bool()

    # ------------------------------------------------------------------

    def filter_beam_paths(
        self,
        beam_mask: torch.Tensor,
        position: int,
        committed_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Narrow the active-path mask after committing a token.

        Keeps only paths whose token at ``position`` equals the committed
        token for that beam.

        Args:
            beam_mask:        ``[B, beam, N]`` bool — current mask.
            position:         depth index that was just committed.
            committed_tokens: ``[B, beam]`` int64 — token chosen per beam.
        Returns:
            ``[B, beam, N]`` bool — filtered mask (monotonically shrinking).
        """
        path_tok = self.path_vectors[:, position]              # [N]
        consistent = path_tok.view(1, 1, -1) == committed_tokens.unsqueeze(-1)
        return beam_mask & consistent
