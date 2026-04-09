#!/usr/bin/env bash
#
# Step 3 — 定向召回: C++ 在线匹配
# ================================
#
# 使用编译好的 C++ CLI 加载 .pt2 模型，为不同用户执行定向匹配。
# 用户属性在 Python 侧编码为张量文件，C++ 仅做通用推理。
#
# 运行方式 (需先跑 01_build_targeting.py + 编译 C++):
#     bash examples/03_targeting_cpp.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

CLI="$ROOT_DIR/inference_engine/build/torch_recall_cli"
MODEL="$SCRIPT_DIR/output/targeting_model.pt2"
META="$SCRIPT_DIR/output/targeting_meta.json"
TENSORS="$SCRIPT_DIR/output/targeting_tensors.pt"

if [ ! -f "$CLI" ]; then
    echo "错误: C++ CLI 未编译。请先运行:"
    echo "  cd inference_engine && mkdir -p build && cd build"
    echo "  cmake -DCMAKE_PREFIX_PATH=\"\$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')\" .."
    echo "  cmake --build . --config Release"
    exit 1
fi

if [ ! -f "$MODEL" ] || [ ! -f "$META" ]; then
    echo "错误: 模型或元数据文件不存在。请先运行:"
    echo "  PYTHONPATH=index_model python examples/01_build_targeting.py"
    exit 1
fi

echo "======================================================================"
echo "定向召回 C++ 推理演示"
echo "======================================================================"

users=(
    '{"city":"北京","gender":"男","age":30,"price":50.0,"tags":"游戏 科技"}'
    '{"city":"上海","gender":"女","age":20,"price":80.0,"tags":"美食 旅行"}'
    '{"city":"广州","gender":"女","age":35,"price":200.0,"tags":"教育"}'
    '{"city":"北京","gender":"女","age":16,"price":30.0,"tags":"游戏 美食"}'
    '{"city":"深圳","gender":"男","age":40,"price":150.0,"tags":"科技"}'
)

for u in "${users[@]}"; do
    echo ""
    echo "----------------------------------------------------------------------"
    echo "用户: $u"
    echo "----------------------------------------------------------------------"

    PYTHONPATH="$ROOT_DIR/index_model" python3 -m torch_recall encode-user --user "$u" --meta "$META" --output "$TENSORS"
    "$CLI" "$MODEL" "$TENSORS"
done

echo ""
echo "======================================================================"
echo "完成！"
