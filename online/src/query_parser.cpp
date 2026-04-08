#include "torch_recall/query_parser.h"
#include <regex>
#include <stdexcept>

namespace torch_recall {

enum class TokenType { KW, OP, STR, LPAREN, RPAREN, WORD };

struct Token {
    TokenType type;
    std::string value;
};

static std::vector<Token> tokenize(const std::string& input) {
    std::vector<Token> tokens;
    std::regex re(
        R"re(\s*(?:(AND|OR|NOT|contains)|([<>!=]=?|==)|"([^"]*)"|'([^']*)'|(\()|(\))|([^\s()'"<>!=]+))\s*)re"
    );
    auto begin = std::sregex_iterator(input.begin(), input.end(), re);
    auto end_it = std::sregex_iterator();
    for (auto it = begin; it != end_it; ++it) {
        auto& m = *it;
        if (m[1].matched) tokens.push_back({TokenType::KW, m[1].str()});
        else if (m[2].matched) tokens.push_back({TokenType::OP, m[2].str()});
        else if (m[3].matched) tokens.push_back({TokenType::STR, m[3].str()});
        else if (m[4].matched) tokens.push_back({TokenType::STR, m[4].str()});
        else if (m[5].matched) tokens.push_back({TokenType::LPAREN, "("});
        else if (m[6].matched) tokens.push_back({TokenType::RPAREN, ")"});
        else if (m[7].matched) tokens.push_back({TokenType::WORD, m[7].str()});
    }
    return tokens;
}

class Parser {
public:
    explicit Parser(std::vector<Token> tokens) : tokens_(std::move(tokens)) {}

    Expr parse() {
        auto result = parse_or();
        if (pos_ != tokens_.size()) {
            throw std::runtime_error("Unexpected tokens at position " + std::to_string(pos_));
        }
        return result;
    }

private:
    std::vector<Token> tokens_;
    size_t pos_ = 0;

    const Token* peek() const {
        return pos_ < tokens_.size() ? &tokens_[pos_] : nullptr;
    }

    Token consume() { return tokens_[pos_++]; }

    bool match_kw(const std::string& kw) const {
        auto* t = peek();
        return t && t->type == TokenType::KW && t->value == kw;
    }

    Expr parse_or() {
        auto left = parse_and();
        std::vector<Expr> children = {std::move(left)};
        while (match_kw("OR")) {
            consume();
            children.push_back(parse_and());
        }
        if (children.size() == 1) return std::move(children[0]);
        return std::make_shared<OrExpr>(OrExpr{std::move(children)});
    }

    Expr parse_and() {
        auto left = parse_not();
        std::vector<Expr> children = {std::move(left)};
        while (match_kw("AND")) {
            consume();
            children.push_back(parse_not());
        }
        if (children.size() == 1) return std::move(children[0]);
        return std::make_shared<AndExpr>(AndExpr{std::move(children)});
    }

    Expr parse_not() {
        if (match_kw("NOT")) {
            consume();
            return std::make_shared<NotExpr>(NotExpr{parse_not()});
        }
        return parse_primary();
    }

    Expr parse_primary() {
        auto* t = peek();
        if (t && t->type == TokenType::LPAREN) {
            consume();
            auto expr = parse_or();
            if (!peek() || peek()->type != TokenType::RPAREN) {
                throw std::runtime_error("Expected ')'");
            }
            consume();
            return expr;
        }
        return parse_predicate();
    }

    Expr parse_predicate() {
        auto field_tok = consume();
        if (field_tok.type != TokenType::WORD) {
            throw std::runtime_error("Expected field name, got: " + field_tok.value);
        }

        if (match_kw("contains")) {
            consume();
            auto val = consume();
            return Predicate{field_tok.value, "contains", val.value};
        }

        auto op_tok = consume();
        auto val_tok = consume();

        std::variant<std::string, double> value;
        if (val_tok.type == TokenType::STR) {
            value = val_tok.value;
        } else {
            try {
                value = std::stod(val_tok.value);
            } catch (...) {
                value = val_tok.value;
            }
        }
        return Predicate{field_tok.value, op_tok.value, value};
    }
};

Expr parse_expression(const std::string& input) {
    auto tokens = tokenize(input);
    Parser parser(std::move(tokens));
    return parser.parse();
}

}  // namespace torch_recall
