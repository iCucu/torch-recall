#include "torch_recall/index_meta.h"
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace torch_recall {

IndexMetadata IndexMetadata::load(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open metadata file: " + json_path);
    }
    nlohmann::json j;
    f >> j;

    IndexMetadata meta;

    for (auto& v : j["schema"]["discrete"]) meta.discrete_fields.push_back(v.get<std::string>());
    for (auto& v : j["schema"]["numeric"]) meta.numeric_fields.push_back(v.get<std::string>());
    for (auto& v : j["schema"]["text"]) meta.text_fields.push_back(v.get<std::string>());

    for (auto& name : meta.discrete_fields) meta.discrete_field_set.insert(name);
    for (auto& name : meta.text_fields) meta.text_field_set.insert(name);
    for (int i = 0; i < static_cast<int>(meta.numeric_fields.size()); i++) {
        meta.numeric_field_index[meta.numeric_fields[i]] = i;
    }

    for (auto& [field, dict] : j["discrete_dicts"].items()) {
        for (auto& [val, id] : dict.items()) {
            meta.discrete_dicts[field][val] = id.get<int>();
        }
    }
    for (auto& [field, dict] : j["text_dicts"].items()) {
        for (auto& [term, id] : dict.items()) {
            meta.text_dicts[field][term] = id.get<int>();
        }
    }
    for (auto& [key, dict] : j["bitmap_lookup"].items()) {
        for (auto& [id_str, gidx] : dict.items()) {
            meta.bitmap_lookup[key][id_str] = gidx.get<int>();
        }
    }

    meta.num_items = j["num_items"].get<int64_t>();
    meta.bitmap_len = j["bitmap_len"].get<int64_t>();
    meta.num_bitmaps = j["num_bitmaps"].get<int64_t>();

    return meta;
}

}  // namespace torch_recall
