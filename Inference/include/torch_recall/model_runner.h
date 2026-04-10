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

    torch::Tensor run(const std::vector<torch::Tensor>& inputs);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace torch_recall
