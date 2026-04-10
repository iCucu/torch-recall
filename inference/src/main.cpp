#include <iostream>
#include <chrono>
#include <string>
#include <torch/torch.h>

#include "torch_recall/model_runner.h"
#include "torch_recall/result_decoder.h"

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <model.pt2> <inputs.pt> [--num-items N]" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string inputs_path = argv[2];
    int64_t num_items = -1;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--num-items" && i + 1 < argc) {
            num_items = std::stoll(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        c10::InferenceMode guard;

        std::cout << "Loading model..." << std::endl;
        torch_recall::ModelRunner runner(model_path);

        std::cout << "Loading inputs..." << std::endl;
        auto loaded = torch::pickle_load(inputs_path);
        auto batches = loaded.toList();
        std::cout << "  Forward passes: " << batches.size() << std::endl;

        std::cout << "Running inference..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        torch::Tensor result;
        for (size_t i = 0; i < batches.size(); ++i) {
            auto tensor_list = batches.get(i).toList();
            std::vector<torch::Tensor> inputs;
            inputs.reserve(tensor_list.size());
            for (size_t j = 0; j < tensor_list.size(); ++j) {
                inputs.push_back(tensor_list.get(j).toTensor());
            }
            auto pass_result = runner.run(inputs);
            if (i == 0) {
                result = pass_result;
            } else {
                result = result | pass_result;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        if (num_items > 0) {
            torch_recall::ResultDecoder decoder(num_items);
            auto ids = decoder.decode(result);

            std::cout << "Results:" << std::endl;
            std::cout << "  Matching items: " << ids.size() << std::endl;
            std::cout << "  Inference time: " << ms << " ms" << std::endl;

            if (ids.size() <= 20) {
                std::cout << "  IDs:";
                for (auto id : ids) std::cout << " " << id;
                std::cout << std::endl;
            } else {
                std::cout << "  First 20 IDs:";
                for (size_t k = 0; k < 20; k++) std::cout << " " << ids[k];
                std::cout << " ..." << std::endl;
            }
        } else {
            std::cout << "Results:" << std::endl;
            std::cout << "  Output shape: " << result.sizes() << std::endl;
            std::cout << "  Output dtype: " << result.dtype() << std::endl;
            std::cout << "  Inference time: " << ms << " ms" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
