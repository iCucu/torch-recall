#pragma once

#include "torch_recall/query_parser.h"
#include "torch_recall/common.h"
#include <vector>

namespace torch_recall {

struct LiteralPred {
    Predicate pred;
    bool negated = false;
};

using Conjunction = std::vector<LiteralPred>;
using DNF = std::vector<Conjunction>;

DNF to_dnf(const Expr& expr);

}  // namespace torch_recall
