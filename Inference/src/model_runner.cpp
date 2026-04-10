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

torch::Tensor ModelRunner::run(const std::vector<torch::Tensor>& inputs) {
    auto outputs = impl_->loader.run(inputs);
    return outputs[0];
}

}  // namespace torch_recall
