import torch
import torch.nn as nn

from torch_recall.schema import MAX_BP, MAX_NP, MAX_CONJ, P_TOTAL


class InvertedIndexModel(nn.Module):
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
        bitmap_indices: torch.Tensor,
        bitmap_valid: torch.Tensor,
        numeric_fields: torch.Tensor,
        numeric_ops: torch.Tensor,
        numeric_values: torch.Tensor,
        numeric_valid: torch.Tensor,
        negation_mask: torch.Tensor,
        conj_matrix: torch.Tensor,
        conj_valid: torch.Tensor,
    ) -> torch.Tensor:
        L = self.bitmap_len

        bp = self.bitmaps[bitmap_indices]

        cols = self.numeric_data[numeric_fields]
        vals = numeric_values.unsqueeze(1)
        ops = numeric_ops

        numeric_bool = (
            ((cols == vals) & (ops == 0).unsqueeze(1))
            | ((cols < vals) & (ops == 1).unsqueeze(1))
            | ((cols > vals) & (ops == 2).unsqueeze(1))
            | ((cols <= vals) & (ops == 3).unsqueeze(1))
            | ((cols >= vals) & (ops == 4).unsqueeze(1))
        )

        np_bitmap = self._pack_bool_to_bitmap(numeric_bool)

        all_bm = torch.cat([bp, np_bitmap], dim=0)

        neg = negation_mask[:P_TOTAL].unsqueeze(1)
        all_bm = torch.where(neg, (all_bm ^ self.all_ones) & self.valid_mask, all_bm)

        all_valid = torch.cat([bitmap_valid, numeric_valid])
        all_bm = torch.where(
            all_valid.unsqueeze(1),
            all_bm,
            self.all_ones.unsqueeze(0).expand(P_TOTAL, L),
        )

        pred_parts = all_bm.unbind(0)

        result = torch.zeros(L, dtype=torch.int64, device=all_bm.device)

        for c in range(MAX_CONJ):
            conj_row = conj_matrix[c]
            conj_result = self.all_ones.clone()
            for p in range(P_TOTAL):
                selected = torch.where(conj_row[p], pred_parts[p], self.all_ones)
                conj_result = conj_result & selected
            conj_result = torch.where(
                conj_valid[c], conj_result, torch.zeros_like(conj_result)
            )
            result = result | conj_result

        result = result & self.valid_mask

        return result
