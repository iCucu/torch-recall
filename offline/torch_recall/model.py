import torch
import torch.nn as nn

from torch_recall.schema import (
    MAX_BP, MAX_NP, P_TOTAL,
    CONJ_PER_PASS, CONJ_PASS_LEVELS,
)


class InvertedIndexModel(nn.Module):
    """Bitmap inverted index model.

    The forward() handles CONJ_PER_PASS conjunctions in one pass using
    vectorized AND + tree-reduced OR.  For queries exceeding this limit,
    the caller invokes forward() multiple times and ORs the results.
    """

    def __init__(
        self,
        bitmaps: torch.Tensor,
        numeric_data: torch.Tensor,
        item_ids: torch.Tensor,
        num_items: int,
    ):
        super().__init__()
        L = bitmaps.shape[1]

        self.register_buffer("bitmaps", bitmaps)
        self.register_buffer("numeric_data", numeric_data)
        self.register_buffer("item_ids", item_ids)

        all_ones = torch.full((L,), -1, dtype=torch.int64)
        self.register_buffer("all_ones", all_ones)

        valid_mask = self._make_valid_mask(num_items, L)
        self.register_buffer("valid_mask", valid_mask)

        pack_powers = torch.tensor(
            [1 << i for i in range(63)] + [-(1 << 63)], dtype=torch.int64
        )
        self.register_buffer("pack_powers", pack_powers)

        self.num_items = num_items
        self.bitmap_len = L

    @staticmethod
    def _make_valid_mask(num_items: int, L: int) -> torch.Tensor:
        mask = torch.zeros(L, dtype=torch.int64)
        full_words = num_items // 64
        remainder = num_items % 64
        if full_words > 0:
            mask[:full_words] = -1
        if remainder > 0:
            mask[full_words] = (1 << remainder) - 1
        return mask

    def _pack_bool_to_bitmap(self, bool_tensor: torch.Tensor) -> torch.Tensor:
        B = bool_tensor.shape[0]
        N = bool_tensor.shape[1]
        pad_size = self.bitmap_len * 64 - N
        if pad_size > 0:
            bool_tensor = torch.nn.functional.pad(bool_tensor, (0, pad_size))
        reshaped = bool_tensor.view(B, self.bitmap_len, 64).long()
        packed = (reshaped * self.pack_powers.unsqueeze(0).unsqueeze(0)).sum(dim=2)
        return packed

    def forward(
        self,
        bitmap_indices: torch.Tensor,   # [MAX_BP]
        bitmap_valid: torch.Tensor,     # [MAX_BP]
        numeric_fields: torch.Tensor,   # [MAX_NP]
        numeric_ops: torch.Tensor,      # [MAX_NP]
        numeric_values: torch.Tensor,   # [MAX_NP]
        numeric_valid: torch.Tensor,    # [MAX_NP]
        negation_mask: torch.Tensor,    # [P_TOTAL]
        conj_matrix: torch.Tensor,      # [CONJ_PER_PASS, P_TOTAL]
        conj_valid: torch.Tensor,       # [CONJ_PER_PASS]
    ) -> torch.Tensor:
        L = self.bitmap_len

        # ── Gather predicate bitmaps ─────────────────────────────────────
        bp = self.bitmaps[bitmap_indices]  # [MAX_BP, L]

        cols = self.numeric_data[numeric_fields]  # [MAX_NP, N]
        vals = numeric_values.unsqueeze(1)
        ops = numeric_ops

        numeric_bool = (
            ((cols == vals) & (ops == 0).unsqueeze(1))
            | ((cols < vals) & (ops == 1).unsqueeze(1))
            | ((cols > vals) & (ops == 2).unsqueeze(1))
            | ((cols <= vals) & (ops == 3).unsqueeze(1))
            | ((cols >= vals) & (ops == 4).unsqueeze(1))
        )

        np_bitmap = self._pack_bool_to_bitmap(numeric_bool)  # [MAX_NP, L]

        all_bm = torch.cat([bp, np_bitmap], dim=0)  # [P_TOTAL, L]

        # ── Apply negation ───────────────────────────────────────────────
        neg = negation_mask[:P_TOTAL].unsqueeze(1)
        all_bm = torch.where(neg, (all_bm ^ self.all_ones) & self.valid_mask, all_bm)

        # ── Apply predicate validity (invalid → all_ones = AND identity) ─
        all_valid = torch.cat([bitmap_valid, numeric_valid])
        all_bm = torch.where(
            all_valid.unsqueeze(1),
            all_bm,
            self.all_ones.unsqueeze(0).expand(P_TOTAL, L),
        )

        # ── Vectorized conjunction evaluation ────────────────────────────
        # [CONJ_PER_PASS, L], initialised to AND identity
        conj_results = self.all_ones.unsqueeze(0).expand(CONJ_PER_PASS, L).contiguous()

        # AND loop: iterate over predicates (P_TOTAL iterations, small fixed count).
        for p in range(P_TOTAL):
            mask_p = conj_matrix[:, p].unsqueeze(1)           # [CONJ_PER_PASS, 1]
            pred_bm = all_bm[p].unsqueeze(0)                  # [1, L]
            selected = torch.where(mask_p, pred_bm, self.all_ones.unsqueeze(0))
            conj_results = conj_results & selected

        # Zero out invalid conjunctions (OR identity = 0)
        conj_results = torch.where(
            conj_valid.unsqueeze(1),
            conj_results,
            torch.zeros(1, L, dtype=torch.int64, device=conj_results.device),
        )

        # ── OR tree reduction: log2(CONJ_PER_PASS) steps ────────────────
        cr = conj_results
        size = CONJ_PER_PASS
        for _ in range(CONJ_PASS_LEVELS):
            size = size // 2
            cr = cr.view(size, 2, L)
            cr = cr[:, 0, :] | cr[:, 1, :]
        result = cr.squeeze(0)

        result = result & self.valid_mask

        return result
