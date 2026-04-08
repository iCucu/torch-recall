#pragma once

#include "torch_recall/dnf_converter.h"
#include "torch_recall/index_meta.h"
#include <torch/torch.h>
#include <unordered_map>

namespace torch_recall {

struct QueryTensors {
    torch::Tensor bitmap_indices;
    torch::Tensor bitmap_valid;
    torch::Tensor numeric_fields;
    torch::Tensor numeric_ops;
    torch::Tensor numeric_values;
    torch::Tensor numeric_valid;
    torch::Tensor negation_mask;
    torch::Tensor conj_matrix;
    torch::Tensor conj_valid;
};

/// Encode a DNF into a list of QueryTensors batches.
/// Each batch handles up to CONJ_PER_PASS conjunctions.
std::vector<QueryTensors> encode_dnf(const DNF& dnf, const IndexMetadata& meta);

}  // namespace torch_recall
