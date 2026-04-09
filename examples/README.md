# examples — 定向召回 完整演示

## 流程概览

```
01_build_targeting.py        02_query_targeting.py        03_targeting_cpp.sh
      │                            │                            │
      ▼                            ▼                            ▼
┌─────────────┐           ┌──────────────┐           ┌──────────────┐
│  定义 Schema │           │  重建模型     │           │  Python 编码  │
│  定义 Rules  │           │  加载 meta   │           │  encode_user  │
│  构建索引    │──output──▶│  encode_user │           │     ↓         │
│  导出 .pt2   │           │  model.query │           │  C++ 加载 .pt2│
│  保存 meta   │           │  解码结果    │           │  + tensors.pt │
└─────────────┘           └──────────────┘           │  通用推理     │
    (Python)                 (Python)                └──────────────┘
                                                      (Python + C++)
```

## 运行步骤

```bash
# 0. 确保已安装依赖
cd torch-recall
source .venv/bin/activate

# 1. 离线: 构建定向索引 + 导出模型
PYTHONPATH=index_model python examples/01_build_targeting.py

# 2. 在线: Python 推理
PYTHONPATH=index_model python examples/02_query_targeting.py

# 3. 在线: C++ 推理 (需先编译)
bash examples/03_targeting_cpp.sh
```

## 产出文件

运行 step 1 后会在 `examples/output/` 下生成:

| 文件 | 说明 |
|---|---|
| `targeting_model.pt2` | AOTInductor 编译后的模型包，可被 C++ `AOTIModelPackageLoader` 加载 |
| `targeting_meta.json` | 谓词注册表: 离散/数值/文本谓词映射 + 模型维度信息 |
| `targeting_rules.json` | 原始定向规则，方便查看结果对应的 item |

## C++ 推理引擎

C++ 推理引擎是一个通用的 tensor-in / tensor-out 推理器，不包含任何业务逻辑:

```
torch_recall_cli <model.pt2> <inputs.pt> [--num-items N]
```

- `inputs.pt` — Python 端编码好的张量文件 (`list[list[Tensor]]`)
- `--num-items N` — 如指定，将布尔结果解码为 item IDs

用户属性编码在 Python 侧完成 (`python -m torch_recall encode-user`)。

## 演示数据

8 条 item 定向规则，涵盖三种字段类型:

- **离散字段**: city (北京/上海/广州), gender (男/女)
- **数值字段**: age, price
- **文本字段**: tags (空格分词)

## 演示规则

| 规则 | 说明 |
|---|---|
| `city == "北京" AND gender == "男"` | 定向北京男性 |
| `city == "上海"` | 定向上海用户 |
| `age > 18` | 定向成年用户 |
| `(city == "北京" OR city == "上海") AND age >= 25` | 定向北京/上海 25+ 用户 |
| `tags contains "游戏"` | 定向游戏用户 |
| `price < 100.0 AND tags contains "美食"` | 定向价格敏感美食用户 |
| `city != "广州"` | 定向非广州用户 |
| `(city == "广州" AND gender == "女") OR age > 30` | 定向广州女性或 30+ 用户 |
