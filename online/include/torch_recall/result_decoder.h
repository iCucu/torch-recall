#pragma once

#include <cstdint>
#include <vector>
#include <torch/torch.h>

namespace torch_recall {

class ResultDecoder {
public:
    explicit ResultDecoder(int64_t num_items);

    std::vector<int64_t> decode(const torch::Tensor& packed_bitmap) const;

private:
    int64_t num_items_;
};

}  // namespace torch_recall
