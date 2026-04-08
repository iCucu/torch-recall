#include "torch_recall/tensor_encoder.h"
#include <stdexcept>
#include <algorithm>

namespace torch_recall {

static const std::unordered_map<std::string, int64_t> OP_MAP = {
    {"==", 0}, {"<", 1}, {">", 2}, {"<=", 3}, {">=", 4}
};

std::vector<QueryTensors> encode_dnf(const DNF& dnf, const IndexMetadata& meta) {

    // ── Step 1: Allocate shared predicate tensors ───────────────────────
    auto bitmap_indices = torch::zeros({MAX_BP}, torch::kInt64);
    auto bitmap_valid   = torch::zeros({MAX_BP}, torch::kBool);
    auto numeric_fields = torch::zeros({MAX_NP}, torch::kInt64);
    auto numeric_ops    = torch::zeros({MAX_NP}, torch::kInt64);
    auto numeric_values = torch::zeros({MAX_NP}, torch::kFloat32);
    auto numeric_valid  = torch::zeros({MAX_NP}, torch::kBool);
    auto negation_mask  = torch::zeros({P_TOTAL}, torch::kBool);

    struct PredKey {
        std::string type;
        int64_t id;
        bool neg;
        bool operator==(const PredKey& o) const {
            return type == o.type && id == o.id && neg == o.neg;
        }
    };
    struct PredKeyHash {
        size_t operator()(const PredKey& k) const {
            return std::hash<std::string>()(k.type)
                ^ (std::hash<int64_t>()(k.id) << 1)
                ^ (std::hash<bool>()(k.neg) << 2);
        }
    };

    std::unordered_map<PredKey, int, PredKeyHash> bp_map, np_map;
    int bp_count = 0, np_count = 0;

    auto resolve_bitmap = [&](const Predicate& pred, bool neg) -> int {
        std::string prefix;
        std::string val_str;

        if (meta.discrete_field_set.count(pred.field)) {
            prefix = "d:" + pred.field;
            auto val = std::get<std::string>(pred.value);
            auto dit = meta.discrete_dicts.find(pred.field);
            if (dit == meta.discrete_dicts.end()) return -1;
            auto vit = dit->second.find(val);
            if (vit == dit->second.end()) return -1;
            val_str = std::to_string(vit->second);
        } else if (meta.text_field_set.count(pred.field) && pred.op == "contains") {
            prefix = "t:" + pred.field;
            auto term = std::get<std::string>(pred.value);
            auto tit = meta.text_dicts.find(pred.field);
            if (tit == meta.text_dicts.end()) return -1;
            auto vit = tit->second.find(term);
            if (vit == tit->second.end()) return -1;
            val_str = std::to_string(vit->second);
        } else {
            return -1;
        }

        auto bit = meta.bitmap_lookup.find(prefix);
        if (bit == meta.bitmap_lookup.end()) return -1;
        auto git = bit->second.find(val_str);
        if (git == bit->second.end()) return -1;
        int64_t global_idx = git->second;

        PredKey key{"bitmap", global_idx, neg};
        if (bp_map.count(key)) return bp_map[key];
        if (bp_count >= MAX_BP) throw std::runtime_error("Exceeds MAX_BP");

        int idx = bp_count++;
        bitmap_indices[idx] = global_idx;
        bitmap_valid[idx] = true;
        negation_mask[idx] = neg;
        bp_map[key] = idx;
        return idx;
    };

    auto resolve_numeric = [&](const Predicate& pred, bool neg) -> int {
        auto nit = meta.numeric_field_index.find(pred.field);
        if (nit == meta.numeric_field_index.end()) return -1;
        auto oit = OP_MAP.find(pred.op);
        if (oit == OP_MAP.end()) return -1;
        double val = std::get<double>(pred.value);

        PredKey key{"numeric", static_cast<int64_t>(nit->second * 100 + oit->second), neg};
        if (np_map.count(key)) return np_map[key];
        if (np_count >= MAX_NP) throw std::runtime_error("Exceeds MAX_NP");

        int idx = np_count++;
        numeric_fields[idx] = nit->second;
        numeric_ops[idx] = oit->second;
        numeric_values[idx] = static_cast<float>(val);
        numeric_valid[idx] = true;
        negation_mask[MAX_BP + idx] = neg;
        np_map[key] = idx;
        return idx;
    };

    // First pass: register all predicates so shared tensors are complete
    for (auto& conj : dnf) {
        for (auto& lit : conj) {
            auto pred = lit.pred;
            bool neg = lit.negated;
            if (pred.op == "!=") { pred.op = "=="; neg = !neg; }

            if (meta.discrete_field_set.count(pred.field) ||
                (meta.text_field_set.count(pred.field) && pred.op == "contains")) {
                resolve_bitmap(pred, neg);
            } else if (meta.numeric_field_index.count(pred.field)) {
                resolve_numeric(pred, neg);
            }
        }
    }

    // ── Step 2: Build batched conj_matrix / conj_valid ──────────────────
    std::vector<QueryTensors> batches;
    int total = static_cast<int>(dnf.size());

    for (int start = 0; start < total; start += CONJ_PER_PASS) {
        int end = std::min(start + CONJ_PER_PASS, total);

        auto conj_matrix = torch::zeros({CONJ_PER_PASS, P_TOTAL}, torch::kBool);
        auto conj_valid  = torch::zeros({CONJ_PER_PASS}, torch::kBool);

        for (int ci = 0; ci < end - start; ++ci) {
            conj_valid[ci] = true;
            for (auto& lit : dnf[start + ci]) {
                auto pred = lit.pred;
                bool neg = lit.negated;
                if (pred.op == "!=") { pred.op = "=="; neg = !neg; }

                if (meta.discrete_field_set.count(pred.field) ||
                    (meta.text_field_set.count(pred.field) && pred.op == "contains")) {
                    int idx = resolve_bitmap(pred, neg);
                    if (idx >= 0) conj_matrix[ci][idx] = true;
                } else if (meta.numeric_field_index.count(pred.field)) {
                    int idx = resolve_numeric(pred, neg);
                    if (idx >= 0) conj_matrix[ci][MAX_BP + idx] = true;
                }
            }
        }

        QueryTensors qt;
        qt.bitmap_indices = bitmap_indices;
        qt.bitmap_valid   = bitmap_valid;
        qt.numeric_fields = numeric_fields;
        qt.numeric_ops    = numeric_ops;
        qt.numeric_values = numeric_values;
        qt.numeric_valid  = numeric_valid;
        qt.negation_mask  = negation_mask;
        qt.conj_matrix    = conj_matrix;
        qt.conj_valid     = conj_valid;
        batches.push_back(std::move(qt));
    }

    if (batches.empty()) {
        QueryTensors qt;
        qt.bitmap_indices = bitmap_indices;
        qt.bitmap_valid   = bitmap_valid;
        qt.numeric_fields = numeric_fields;
        qt.numeric_ops    = numeric_ops;
        qt.numeric_values = numeric_values;
        qt.numeric_valid  = numeric_valid;
        qt.negation_mask  = negation_mask;
        qt.conj_matrix    = torch::zeros({CONJ_PER_PASS, P_TOTAL}, torch::kBool);
        qt.conj_valid     = torch::zeros({CONJ_PER_PASS}, torch::kBool);
        batches.push_back(std::move(qt));
    }

    return batches;
}

}  // namespace torch_recall
