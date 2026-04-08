#include "torch_recall/result_decoder.h"

namespace torch_recall {

ResultDecoder::ResultDecoder(int64_t num_items) : num_items_(num_items) {}

std::vector<int64_t> ResultDecoder::decode(const torch::Tensor& packed_bitmap) const {
    auto accessor = packed_bitmap.accessor<int64_t, 1>();
    int64_t L = accessor.size(0);
    std::vector<int64_t> result;
    result.reserve(1024);

    for (int64_t w = 0; w < L; ++w) {
        int64_t word = accessor[w];
        if (word == 0) continue;
        int64_t base = w * 64;
        while (word != 0) {
            int bit = __builtin_ctzll(static_cast<unsigned long long>(word));
            int64_t item_idx = base + bit;
            if (item_idx < num_items_) {
                result.push_back(item_idx);
            }
            word &= word - 1;
        }
    }
    return result;
}

}  // namespace torch_recall
