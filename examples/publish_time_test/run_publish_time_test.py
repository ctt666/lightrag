"""
Test script for publish_time: insert doc A (older), then doc B (newer, conflicting),
inspect entity/relation/chunk structure, and query to verify time-aware merge.

Supports:
- OpenAI: set OPENAI_API_KEY (optionally LLM_MODEL, default gpt-4o-mini).
- Qwen (OpenAI-compatible): set OPENAI_API_KEY, BASE_URL, LLM_MODEL.
  Example: BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 LLM_MODEL=qwen-plus
  Embedding uses same base_url by default; set EMBEDDING_MODEL/EMBEDDING_DIM if needed.
"""
import asyncio
import json
import os
import sys
from functools import partial

# Add project root for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import (
    openai_complete_if_cache,
    openai_embed,
    gpt_4o_mini_complete,
)
from lightrag.utils import EmbeddingFunc
from lightrag.constants import GRAPH_FIELD_SEP

WORKING_DIR = os.path.join(os.path.dirname(__file__), "publish_time_rag")
TEST_DIR = os.path.dirname(__file__)
FILE_A = os.path.join(TEST_DIR, "test_a.txt")
FILE_B = os.path.join(TEST_DIR, "test_b.txt")

# Publish times: A older, B newer (so B should win on conflict)
PUBLISH_TIME_A = "2023-01-01T00:00:00Z"
PUBLISH_TIME_B = "2024-06-01T00:00:00Z"

# Qwen / OpenAI-compatible: read from env
BASE_URL = os.getenv("BASE_URL") or os.getenv("LLM_BINDING_HOST")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
MAX_EMBED_TOKENS = int(os.getenv("MAX_EMBED_TOKENS", "8192"))
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_BINDING_API_KEY") or os.getenv("QWEN_API_KEY")


def _make_llm_func():
    """LLM 调用：若配置了 BASE_URL 则用 Qwen/兼容接口，否则用默认 OpenAI。"""
    if BASE_URL:
        async def qwen_complete(prompt, system_prompt=None, history_messages=None, **kwargs):
            return await openai_complete_if_cache(
                LLM_MODEL,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                base_url=BASE_URL,
                api_key=API_KEY,
                **kwargs,
            )
        return qwen_complete
    return gpt_4o_mini_complete


def _make_embedding_func():
    """Embedding：若配置了 BASE_URL 则用同一 base_url（Qwen 兼容），否则用默认 openai_embed。"""
    if BASE_URL:
        return EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=MAX_EMBED_TOKENS,
            func=partial(
                openai_embed.func,
                model=EMBEDDING_MODEL,
                base_url=BASE_URL,
                api_key=API_KEY,
            ),
        )
    return openai_embed


def _redact_for_display(obj, max_len=200):
    """Truncate long strings for readable print."""
    if isinstance(obj, dict):
        return {k: _redact_for_display(v, max_len) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > 5:
            return [_redact_for_display(x, max_len) for x in obj[:5]] + ["..."]
        return [_redact_for_display(x, max_len) for x in obj]
    if isinstance(obj, str) and len(obj) > max_len:
        return obj[:max_len] + "..."
    return obj


async def print_structure(rag, stage: str):
    """Print entity, relation, and sample chunk structure (including publish_time)."""
    print(f"\n{'='*60}")
    print(f" 结构快照: {stage}")
    print(f"{'='*60}")

    # Entities
    nodes = await rag.chunk_entity_relation_graph.get_all_nodes()
    print(f"\n[Entity 数量]: {len(nodes)}")
    for i, n in enumerate(nodes[:5]):
        d = _redact_for_display(dict(n))
        print(f"  Entity[{i}]: {json.dumps(d, ensure_ascii=False, indent=2)}")
    if len(nodes) > 5:
        print(f"  ... 共 {len(nodes)} 个 entity")

    # Relations
    edges = await rag.chunk_entity_relation_graph.get_all_edges()
    print(f"\n[Relation 数量]: {len(edges)}")
    for i, e in enumerate(edges[:5]):
        d = _redact_for_display(dict(e))
        print(f"  Relation[{i}]: {json.dumps(d, ensure_ascii=False, indent=2)}")
    if len(edges) > 5:
        print(f"  ... 共 {len(edges)} 条 relation")

    # Sample chunk (from first entity's source_id if any)
    if nodes:
        first_node = nodes[0]
        source_id = first_node.get("source_id") or ""
        chunk_id = source_id.split(GRAPH_FIELD_SEP)[0].strip() if source_id else None
        if chunk_id:
            chunk = await rag.text_chunks.get_by_id(chunk_id)
            if chunk:
                d = _redact_for_display(dict(chunk))
                print(f"\n[Chunk 示例] id={chunk_id}")
                print(f"  {json.dumps(d, ensure_ascii=False, indent=2)}")
    print()


async def main():
    if not API_KEY:
        print("请设置 OPENAI_API_KEY（或 QWEN_API_KEY）后重试")
        return

    os.makedirs(WORKING_DIR, exist_ok=True)

    if BASE_URL:
        print(f"使用 Qwen/兼容接口: BASE_URL={BASE_URL!r} LLM_MODEL={LLM_MODEL!r} EMBEDDING_MODEL={EMBEDDING_MODEL!r}\n")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=_make_embedding_func(),
        llm_model_func=_make_llm_func(),
    )
    await rag.initialize_storages()

    try:
        # 1) 读入并导入 A（旧）
        # with open(FILE_A, "r", encoding="utf-8") as f:
        #     content_a = f.read()
        # print(f"[1] 导入 A (publish_time={PUBLISH_TIME_A})")
        # await rag.ainsert(content_a, publish_time=PUBLISH_TIME_A)
        # await print_structure(rag, "导入 A 之后")

        # # 2) 读入并导入 B（新，与 A 有矛盾）
        # with open(FILE_B, "r", encoding="utf-8") as f:
        #     content_b = f.read()
        # print(f"[2] 导入 B (publish_time={PUBLISH_TIME_B})")
        # await rag.ainsert(content_b, publish_time=PUBLISH_TIME_B)
        # await print_structure(rag, "导入 B 之后")

        # # 3) 提问验证：应偏向 B（成立时间 2005、总部上海、产品云计算）
        # print(f"\n{'='*60}")
        # print(" 提问验证（应体现较新资料 B：2005年、上海、云计算）")
        # print(f"{'='*60}\n")
        query = "星海科技何时成立？总部在哪里？主要产品是什么？"
        result = await rag.aquery(query, param=QueryParam(mode="global"))
        print(f"问: {query}")
        print(f"答: {result}\n")
    finally:
        await rag.finalize_storages()

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
