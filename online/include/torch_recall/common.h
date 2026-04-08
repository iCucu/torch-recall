#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

namespace torch_recall {

constexpr int MAX_BP = 16;
constexpr int MAX_NP = 8;
constexpr int MAX_CONJ = 1024;    // system-level limit
constexpr int CONJ_PER_PASS = 8;  // model-level limit per forward pass
constexpr int P_TOTAL = MAX_BP + MAX_NP;

enum class NumericOp : int64_t {
    EQ = 0, LT = 1, GT = 2, LE = 3, GE = 4
};

}  // namespace torch_recall
