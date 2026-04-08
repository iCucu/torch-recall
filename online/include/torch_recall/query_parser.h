#pragma once

#include <string>
#include <memory>
#include <vector>
#include <variant>

namespace torch_recall {

struct Predicate {
    std::string field;
    std::string op;
    std::variant<std::string, double> value;
};

struct AndExpr;
struct OrExpr;
struct NotExpr;

using Expr = std::variant<
    Predicate,
    std::shared_ptr<AndExpr>,
    std::shared_ptr<OrExpr>,
    std::shared_ptr<NotExpr>
>;

struct AndExpr { std::vector<Expr> children; };
struct OrExpr  { std::vector<Expr> children; };
struct NotExpr { Expr child; };

Expr parse_expression(const std::string& input);

}  // namespace torch_recall
