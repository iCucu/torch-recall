#include "torch_recall/model_runner.h"
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

namespace torch_recall {

struct ModelRunner::Impl {
    torch::inductor::AOTIModelPackageLoader loader;
    explicit Impl(const std::string& path) : loader(path) {}
};

ModelRunner::ModelRunner(const std::string& pt2_path)
    : impl_(std::make_unique<Impl>(pt2_path)) {}

ModelRunner::~ModelRunner() = default;

torch::Tensor ModelRunner::run(
    torch::Tensor bitmap_indices,
    torch::Tensor bitmap_valid,
    torch::Tensor numeric_fields,
    torch::Tensor numeric_ops,
    torch::Tensor numeric_values,
    torch::Tensor numeric_valid,
    torch::Tensor negation_mask,
    torch::Tensor conj_matrix,
    torch::Tensor conj_valid
) {
    std::vector<torch::Tensor> inputs = {
        bitmap_indices, bitmap_valid,
        numeric_fields, numeric_ops, numeric_values, numeric_valid,
        negation_mask, conj_matrix, conj_valid
    };
    auto outputs = impl_->loader.run(inputs);
    return outputs[0];
}

}  // namespace torch_recall
