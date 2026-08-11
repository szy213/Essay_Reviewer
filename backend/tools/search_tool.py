"""Search tool for RAG knowledge base — hello_agents Tool interface."""

import sys
from pathlib import Path
from typing import Any, Dict, List

from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.response import ToolResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_base.vector_store import VectorStore


class SearchTool(Tool):
    """Semantic search over the personal knowledge base."""

    def __init__(self, collection_name: str = "maogai_knowledge", persist_dir: str = "data/chroma_db"):
        super().__init__(
            name="search_knowledge_base",
            description=(
                "在个人知识库中语义搜索相关内容。"
                "当用户询问知识库中的知识点、概念、理论时使用此工具。"
                "返回最相关的文本片段及其来源页码。"
            ),
        )
        persist_path = PROJECT_ROOT / persist_dir
        self._store = VectorStore(
            collection_name=collection_name,
            persist_dir=str(persist_path),
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="搜索查询，使用中文关键词或自然语言问题",
                required=True,
            ),
            ToolParameter(
                name="top_k",
                type="integer",
                description="返回结果数量，默认 5",
                required=False,
                default=5,
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        query = parameters["query"]
        top_k = parameters.get("top_k", 5)
        results = self._store.search(query, top_k=top_k)

        if not results:
            return ToolResponse.success(text="知识库中没有找到相关内容。")

        lines = [f"搜索「{query}」返回 {len(results)} 条结果：\n"]
        for i, r in enumerate(results):
            page = r["metadata"]["page"] + 1
            filename = r["metadata"].get("filename", "未知")
            score = r["score"]
            text = r["text"].replace("\n", " ")
            lines.append(
                f"[{i + 1}] (相关度: {score:.2%}, 来源: {filename} 第{page}页)\n"
                f"    {text[:300]}"
            )

        return ToolResponse.success(text="\n\n".join(lines))
