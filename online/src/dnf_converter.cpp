#include "torch_recall/dnf_converter.h"
#include <stdexcept>

namespace torch_recall {

static DNF convert(const Expr& expr);
static DNF negate(const Expr& expr);

static DNF convert(const Expr& expr) {
    if (auto* p = std::get_if<Predicate>(&expr)) {
        return {{LiteralPred{*p, false}}};
    }
    if (auto* n = std::get_if<std::shared_ptr<NotExpr>>(&expr)) {
        return negate((*n)->child);
    }
    if (auto* a = std::get_if<std::shared_ptr<AndExpr>>(&expr)) {
        DNF result = {{}};
        for (auto& child : (*a)->children) {
            auto child_dnf = convert(child);
            DNF new_result;
            for (auto& existing : result) {
                for (auto& child_conj : child_dnf) {
                    auto merged = existing;
                    merged.insert(merged.end(), child_conj.begin(), child_conj.end());
                    new_result.push_back(std::move(merged));
                }
            }
            result = std::move(new_result);
            if (static_cast<int>(result.size()) > MAX_CONJ) {
                throw std::runtime_error("DNF exceeds MAX_CONJ. Simplify the query.");
            }
        }
        return result;
    }
    if (auto* o = std::get_if<std::shared_ptr<OrExpr>>(&expr)) {
        DNF result;
        for (auto& child : (*o)->children) {
            auto child_dnf = convert(child);
            result.insert(result.end(), child_dnf.begin(), child_dnf.end());
            if (static_cast<int>(result.size()) > MAX_CONJ) {
                throw std::runtime_error("DNF exceeds MAX_CONJ. Simplify the query.");
            }
        }
        return result;
    }
    throw std::runtime_error("Unknown expression type");
}

static DNF negate(const Expr& expr) {
    if (auto* p = std::get_if<Predicate>(&expr)) {
        return {{LiteralPred{*p, true}}};
    }
    if (auto* n = std::get_if<std::shared_ptr<NotExpr>>(&expr)) {
        return convert((*n)->child);
    }
    if (auto* a = std::get_if<std::shared_ptr<AndExpr>>(&expr)) {
        std::vector<Expr> negated_children;
        for (auto& c : (*a)->children) {
            negated_children.push_back(std::make_shared<NotExpr>(NotExpr{c}));
        }
        auto or_expr = std::make_shared<OrExpr>(OrExpr{std::move(negated_children)});
        return convert(or_expr);
    }
    if (auto* o = std::get_if<std::shared_ptr<OrExpr>>(&expr)) {
        std::vector<Expr> negated_children;
        for (auto& c : (*o)->children) {
            negated_children.push_back(std::make_shared<NotExpr>(NotExpr{c}));
        }
        auto and_expr = std::make_shared<AndExpr>(AndExpr{std::move(negated_children)});
        return convert(and_expr);
    }
    throw std::runtime_error("Unknown expression type in negate");
}

DNF to_dnf(const Expr& expr) {
    return convert(expr);
}

}  // namespace torch_recall
