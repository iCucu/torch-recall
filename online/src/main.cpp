#include <iostream>
#include <chrono>
#include <torch/torch.h>

#include "torch_recall/index_meta.h"
#include "torch_recall/model_runner.h"
#include "torch_recall/query_parser.h"
#include "torch_recall/dnf_converter.h"
#include "torch_recall/tensor_encoder.h"
#include "torch_recall/result_decoder.h"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.pt2> <index_meta.json> <query>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string meta_path = argv[2];
    std::string query_str = argv[3];

    try {
        c10::InferenceMode guard;

        std::cout << "Loading metadata..." << std::endl;
        auto meta = torch_recall::IndexMetadata::load(meta_path);

        std::cout << "Loading model..." << std::endl;
        torch_recall::ModelRunner runner(model_path);

        std::cout << "Parsing query: " << query_str << std::endl;
        auto expr = torch_recall::parse_expression(query_str);
        auto dnf = torch_recall::to_dnf(expr);
        std::cout << "  DNF conjunctions: " << dnf.size() << std::endl;

        auto qt = torch_recall::encode_dnf(dnf, meta);

        std::cout << "Running query..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        auto result = runner.run(
            qt.bitmap_indices, qt.bitmap_valid,
            qt.numeric_fields, qt.numeric_ops,
            qt.numeric_values, qt.numeric_valid,
            qt.negation_mask, qt.conj_matrix, qt.conj_valid
        );

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        torch_recall::ResultDecoder decoder(meta.num_items);
        auto ids = decoder.decode(result);

        std::cout << "Results:" << std::endl;
        std::cout << "  Matching items: " << ids.size() << std::endl;
        std::cout << "  Query time: " << ms << " ms" << std::endl;

        if (ids.size() <= 20) {
            std::cout << "  IDs:";
            for (auto id : ids) std::cout << " " << id;
            std::cout << std::endl;
        } else {
            std::cout << "  First 20 IDs:";
            for (size_t i = 0; i < 20; i++) std::cout << " " << ids[i];
            std::cout << " ..." << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
