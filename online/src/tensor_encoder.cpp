#include "torch_recall/tensor_encoder.h"
#include <stdexcept>

namespace torch_recall {

static const std::unordered_map<std::string, int64_t> OP_MAP = {
    {"==", 0}, {"<", 1}, {">", 2}, {"<=", 3}, {">=", 4}
};

QueryTensors encode_dnf(const DNF& dnf, const IndexMetadata& meta) {
    QueryTensors qt;
    qt.bitmap_indices = torch::zeros({MAX_BP}, torch::kInt64);
    qt.bitmap_valid = torch::zeros({MAX_BP}, torch::kBool);
    qt.numeric_fields = torch::zeros({MAX_NP}, torch::kInt64);
    qt.numeric_ops = torch::zeros({MAX_NP}, torch::kInt64);
    qt.numeric_values = torch::zeros({MAX_NP}, torch::kFloat32);
    qt.numeric_valid = torch::zeros({MAX_NP}, torch::kBool);
    qt.negation_mask = torch::zeros({P_TOTAL}, torch::kBool);
    qt.conj_matrix = torch::zeros({MAX_CONJ, P_TOTAL}, torch::kBool);
    qt.conj_valid = torch::zeros({MAX_CONJ}, torch::kBool);

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
        qt.bitmap_indices[idx] = global_idx;
        qt.bitmap_valid[idx] = true;
        qt.negation_mask[idx] = neg;
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
        qt.numeric_fields[idx] = nit->second;
        qt.numeric_ops[idx] = oit->second;
        qt.numeric_values[idx] = static_cast<float>(val);
        qt.numeric_valid[idx] = true;
        qt.negation_mask[MAX_BP + idx] = neg;
        np_map[key] = idx;
        return idx;
    };

    for (int ci = 0; ci < static_cast<int>(dnf.size()) && ci < MAX_CONJ; ++ci) {
        qt.conj_valid[ci] = true;
        for (auto& lit : dnf[ci]) {
            auto pred = lit.pred;
            bool neg = lit.negated;

            if (pred.op == "!=") {
                pred.op = "==";
                neg = !neg;
            }

            if (meta.discrete_field_set.count(pred.field) ||
                (meta.text_field_set.count(pred.field) && pred.op == "contains")) {
                int idx = resolve_bitmap(pred, neg);
                if (idx >= 0) qt.conj_matrix[ci][idx] = true;
            } else if (meta.numeric_field_index.count(pred.field)) {
                int idx = resolve_numeric(pred, neg);
                if (idx >= 0) qt.conj_matrix[ci][MAX_BP + idx] = true;
            }
        }
    }

    return qt;
}

}  // namespace torch_recall
