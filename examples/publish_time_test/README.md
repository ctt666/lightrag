# publish_time 测试

- **test_a.txt**: 星海科技 — 1990年成立、北京、办公软件（旧资料）
- **test_b.txt**: 星海科技 — 2005年成立、上海、云计算（新资料，与 A 矛盾）

## 运行

### 使用 OpenAI

在项目根目录执行（需设置 `OPENAI_API_KEY`）：

```bash
cd /path/to/LightRAG
export OPENAI_API_KEY='your-key'
uv run python examples/publish_time_test/run_publish_time_test.py
```

### 使用 Qwen（OpenAI 兼容接口，需配置 base_url）

Qwen / 通义千问等兼容 OpenAI 的接口需额外配置 `BASE_URL` 和模型名：

```bash
cd /path/to/LightRAG
export OPENAI_API_KEY='your-qwen-or-dashscope-key'
export BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
export LLM_MODEL='qwen-plus'
uv run python examples/publish_time_test/run_publish_time_test.py
```

可选环境变量：

- `BASE_URL` 或 `LLM_BINDING_HOST`：兼容接口地址（设置后即走 Qwen/兼容路径）
- `LLM_MODEL`：对话模型，默认 `gpt-4o-mini`，Qwen 可用 `qwen-plus`、`qwen-turbo` 等
- `EMBEDDING_MODEL`：嵌入模型，默认 `text-embedding-3-small`；DashScope 可用 `text-embedding-v3`
- `EMBEDDING_DIM`：嵌入维度，默认 `1536`
- `MAX_EMBED_TOKENS`：嵌入最大 token，默认 `8192`
- `OPENAI_API_KEY` / `LLM_BINDING_API_KEY` / `QWEN_API_KEY`：API Key（任选其一）

或先进入示例目录再执行：

```bash
cd examples/publish_time_test
export OPENAI_API_KEY='your-key'
# 使用 Qwen 时再加上 BASE_URL、LLM_MODEL
python run_publish_time_test.py
```

脚本会：

1. 导入 A（`publish_time=2023-01-01T00:00:00Z`），并打印此时 entity/relation/chunk 结构（含 `publish_time`）
2. 导入 B（`publish_time=2024-06-01T00:00:00Z`），再次打印结构
3. 提问：「星海科技何时成立？总部在哪里？主要产品是什么？」  
   预期答案应偏向 B：2005年、上海、云计算

数据目录：`examples/publish_time_test/publish_time_rag/`（默认 file 存储）。
