#!/usr/bin/env bash
#
# Step 3 — C++ 在线推理
# =====================
#
# 使用编译好的 C++ CLI 加载 .pt2 模型，执行查询。
#
# 运行方式 (需先跑 01_build_index.py + 编译 C++):
#     bash examples/03_query_cpp.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

CLI="$ROOT_DIR/online/build/torch_recall_cli"
MODEL="$SCRIPT_DIR/output/model.pt2"
META="$SCRIPT_DIR/output/index_meta.json"

if [ ! -f "$CLI" ]; then
    echo "错误: C++ CLI 未编译。请先运行:"
    echo "  cd online && mkdir -p build && cd build"
    echo "  cmake -DCMAKE_PREFIX_PATH=\"\$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')\" .."
    echo "  cmake --build . --config Release"
    exit 1
fi

if [ ! -f "$MODEL" ] || [ ! -f "$META" ]; then
    echo "错误: 模型或元数据文件不存在。请先运行:"
    echo "  python examples/01_build_index.py"
    exit 1
fi

echo "======================================================================"
echo "C++ 在线推理演示"
echo "======================================================================"

queries=(
    'city == "北京"'
    'city == "北京" AND gender == "男"'
    'price < 100.0'
    '(city == "北京" OR city == "上海") AND price < 200.0'
    'NOT category == "美食"'
    'title contains "攻略"'
    'title contains "游戏" AND price < 60.0'
    '(title contains "美食" OR category == "教育") AND score >= 4.3'
)

for q in "${queries[@]}"; do
    echo ""
    echo "----------------------------------------------------------------------"
    echo "查询: $q"
    echo "----------------------------------------------------------------------"
    "$CLI" "$MODEL" "$META" "$q"
done

echo ""
echo "======================================================================"
echo "完成！"
