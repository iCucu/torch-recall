#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace torch_recall {

struct IndexMetadata {
    std::vector<std::string> discrete_fields;
    std::vector<std::string> numeric_fields;
    std::vector<std::string> text_fields;

    std::unordered_set<std::string> discrete_field_set;
    std::unordered_set<std::string> text_field_set;
    std::unordered_map<std::string, int> numeric_field_index;

    std::unordered_map<std::string, std::unordered_map<std::string, int>> discrete_dicts;
    std::unordered_map<std::string, std::unordered_map<std::string, int>> text_dicts;
    std::unordered_map<std::string, std::unordered_map<std::string, int>> bitmap_lookup;

    int64_t num_items = 0;
    int64_t bitmap_len = 0;
    int64_t num_bitmaps = 0;

    static IndexMetadata load(const std::string& json_path);
};

}  // namespace torch_recall
