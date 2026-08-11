"""RAG Agent — inherits from hello_agents SimpleAgent, with knowledge-base search capability."""

from pathlib import Path
from dotenv import load_dotenv

from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.core.config import Config

from backend.tools.search_tool import SearchTool

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SYSTEM_PROMPT = """
你是一个个人知识库助手。

你的任务是帮助用户理解自己的知识库。

当用户提出问题时：

1. 判断是否需要查询知识库
2. 如果需要，调用 knowledge_search
3. 根据检索结果回答
4. 不要编造知识库中不存在的内容
5. 如果资料不足，请明确告诉用户
6. 回答时注明参考资料来源
"""


class RAGAgent(SimpleAgent):
    """LLM Agent with knowledge-base search capability, based on SimpleAgent."""

    def __init__(self, collection_name: str = "maogai_knowledge"):
        llm = HelloAgentsLLM()
        tool_registry = ToolRegistry()
        tool_registry.register_tool(SearchTool(collection_name=collection_name))

        config = Config(
            trace_enabled=False,
            skills_enabled=False,
            subagent_enabled=False,
            todowrite_enabled=False,
            devlog_enabled=False,
        )

        super().__init__(
            name="RAGAgent",
            llm=llm,
            system_prompt=SYSTEM_PROMPT,
            config=config,
            tool_registry=tool_registry,
        )
