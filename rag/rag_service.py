from agent.scope_router import OUT_OF_SCOPE_ANSWER, is_out_of_scope_query
from rag.citations import build_sources
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import DashScopeReranker
from rag.vector_store import VectorStoreService
from utils.config_handler import chroma_conf, rag_conf
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# 检索管线的可调参数：mode=vector 时退化成纯向量召回，方便评测做 A/B。
_retrieval_conf = rag_conf.get("retrieval") or {}
_rerank_conf = _retrieval_conf.get("rerank") or {}

# 重排后没有任何候选过阈值时的固定回答：宁可承认没找到，也不让模型凭空调编。
NO_EVIDENCE_ANSWER = (
    "知识库里没有找到和这个问题直接相关的内容，为了避免给你错误信息，我先不作答。"
    "可以换个说法再问一次（比如补上机型、错误码或具体场景），"
    "也可以问我选购、使用、维护保养、故障排查方面的问题。"
)


def _build_reranker() -> DashScopeReranker | None:
    if not _rerank_conf.get("enabled", False):
        return None
    return DashScopeReranker(model=_rerank_conf.get("model", "gte-rerank-v2"))


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.pipeline = HybridRetriever(
            self.vector_store,
            mode=_retrieval_conf.get("mode", "hybrid"),
            vector_candidates=int(_retrieval_conf.get("vector_candidates", 20)),
            bm25_candidates=int(_retrieval_conf.get("bm25_candidates", 20)),
            rrf_k=int(_retrieval_conf.get("rrf_k", 60)),
            reranker=_build_reranker(),
            rerank_candidates=int(_rerank_conf.get("candidates", 20)),
            min_score=float(_rerank_conf.get("min_score", 0.0)),
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
        # 重排阈值把候选全砍掉时说明知识库里确实没有依据，直接给固定回答，
        # 既省一次模型调用，也堵住「拿着不相关资料硬编」的路。
        if not context_docs:
            logger.info(f"[rag] 无证据兜底触发：{query[:40]}")
            return NO_EVIDENCE_ANSWER
        return self.chain.invoke(
            {
                "input": query,
                "context": self.build_context(context_docs),
            }
        )

    def answer_with_citations(self, query: str) -> tuple[str, list[dict], dict]:
        """检索 + 生成，并把引用条目与检索统计一起带回来。

        返回 ``(回答, 引用列表, 检索统计)``。越界拒答不带引用：那条回答一个字
        都没有用到检索结果，界面上再列「参考资料」就是在暗示它有依据。
        无证据兜底同理，此时文档本来就是空的。
        """
        documents, stats = self.retrieve_with_stats(query)
        answer = self.answer(query, documents)
        if answer == OUT_OF_SCOPE_ANSWER:
            return answer, [], stats
        return answer, build_sources(documents), stats


if __name__ == '__main__':
    rag = RagSummarizeService()
    rag.ensure_knowledge_base()
    text, sources, retrieval = rag.answer_with_citations('小户型适合哪些扫地机器人')
    print(text)
    print(f"引用 {len(sources)} 条：{sources}")
    print(f"检索统计：{retrieval}")
