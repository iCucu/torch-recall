# examples — 离线构建 + 在线推理 完整演示

## 流程概览

```
01_build_index.py          02_query_python.py          03_query_cpp.sh
      │                          │                           │
      ▼                          ▼                           ▼
┌─────────────┐           ┌─────────────┐           ┌──────────────┐
│  定义 Schema │           │  加载 meta  │           │  加载 meta   │
│  准备 items  │           │  重建模型   │           │  加载 .pt2   │
│  构建索引    │──output──▶│  encode 查询 │           │  parse 查询   │
│  导出 .pt2   │           │  forward 推理│           │  encode 查询  │
│  保存 meta   │           │  解码结果   │           │  forward 推理 │
└─────────────┘           └─────────────┘           └──────────────┘
    (Python)                 (Python)                    (C++)
```

## 运行步骤

```bash
# 0. 确保已安装依赖
cd torch-recall
source .venv/bin/activate

# 1. 离线: 构建索引 + 导出模型
python examples/01_build_index.py

# 2. 在线: Python 推理
python examples/02_query_python.py

# 3. 在线: C++ 推理 (需先编译)
bash examples/03_query_cpp.sh
```

## 产出文件

运行 step 1 后会在 `examples/output/` 下生成:

| 文件 | 说明 |
|---|---|
| `model.pt2` | AOTInductor 编译后的模型包，可被 C++ `AOTIModelPackageLoader` 加载 |
| `index_meta.json` | 索引元数据: schema、字段字典、bitmap 映射关系 |
| `items.json` | 原始数据，方便查看查询结果对应的 item |

## 演示数据

8 条广告 item，涵盖三种字段类型:

- **离散字段**: city (北京/上海/广州/深圳), gender (男/女), category (游戏/美食/教育/科技)
- **数值字段**: price, score
- **文本字段**: title (空格分词)

## 演示查询

| 查询 | 说明 |
|---|---|
| `city == "北京"` | 离散等值 |
| `city == "北京" AND gender == "男"` | 离散 AND |
| `price < 100.0` | 数值范围 |
| `(city == "北京" OR city == "上海") AND price < 200.0` | OR + 数值 |
| `NOT category == "美食"` | NOT |
| `title contains "攻略"` | 文本匹配 |
| `title contains "游戏" AND price < 60.0` | 文本 + 数值 |
| `(title contains "美食" OR category == "教育") AND score >= 4.3` | 混合复杂查询 |
