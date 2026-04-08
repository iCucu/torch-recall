#pragma once

#include <string>
#include <vector>
#include <memory>
#include <torch/torch.h>

namespace torch_recall {

class ModelRunner {
public:
    explicit ModelRunner(const std::string& pt2_path);
    ~ModelRunner();

    torch::Tensor run(
        torch::Tensor bitmap_indices,
        torch::Tensor bitmap_valid,
        torch::Tensor numeric_fields,
        torch::Tensor numeric_ops,
        torch::Tensor numeric_values,
        torch::Tensor numeric_valid,
        torch::Tensor negation_mask,
        torch::Tensor conj_matrix,
        torch::Tensor conj_valid
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace torch_recall
