from agent.scope_router import OUT_OF_SCOPE_ANSWER, is_out_of_scope_query
from rag.hybrid_retriever import HybridRetriever
from rag.vector_store import VectorStoreService
from utils.config_handler import chroma_conf, rag_conf
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# 检索管线的可调参数：mode=vector 时退化成纯向量召回，方便评测做 A/B。
_retrieval_conf = rag_conf.get("retrieval") or {}


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.pipeline = HybridRetriever(
            self.vector_store,
            mode=_retrieval_conf.get("mode", "hybrid"),
            vector_candidates=int(_retrieval_conf.get("vector_candidates", 20)),
            bm25_candidates=int(_retrieval_conf.get("bm25_candidates", 20)),
            rrf_k=int(_retrieval_conf.get("rrf_k", 60)),
        )
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def ensure_knowledge_base(self) -> int:
        """启动时检查向量库；空库自动重建，有数据则同步变更，最后刷新 BM25 索引。"""
        count = self.vector_store.ensure_knowledge_base()
        self.pipeline.refresh()
        return count

    def retrieve_with_stats(self, query: str, k: int | None = None) -> tuple[list[Document], dict]:
        """检索并附带召回统计（各路候选数、融合数、延迟），供观测和引用溯源使用。"""
        limit = k if k is not None else chroma_conf["k"]
        result = self.pipeline.search(query, limit)
        return result.documents, result.stats

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        """只负责检索，返回原始 Document，供评测和上层回答复用。"""
        return self.retrieve_with_stats(query, k)[0]

    def build_context(self, context_docs: list[Document]) -> str:
        """把检索结果拼成提示词里的参考资料区，保持原有编号格式。"""
        parts = []
        for counter, doc in enumerate(context_docs, start=1):
            parts.append(
                f"【参考资料{counter}】: {doc.page_content} | 元数据: {doc.metadata}\n"
            )
        return "".join(parts)

    def answer(self, query: str, context_docs: list[Document]) -> str:
        """只负责生成；检索结果由调用方传入，便于单独评测召回阶段。"""
        # 越界请求在进入模型前直接拒答，避免模型顺手完成编程/翻译等任务。
        if is_out_of_scope_query(query):
            return OUT_OF_SCOPE_ANSWER
        return self.chain.invoke(
            {
                "input": query,
                "context": self.build_context(context_docs),
            }
        )

    def rag_summarize(self, query: str) -> str:
        """检索 + 生成的一条龙入口，供 Agent 工具继续使用。"""
        return self.answer(query, self.retrieve(query))


if __name__ == '__main__':
    rag = RagSummarizeService()
    rag.ensure_knowledge_base()
    print(rag.rag_summarize('小户型适合哪些扫地机器人'))
