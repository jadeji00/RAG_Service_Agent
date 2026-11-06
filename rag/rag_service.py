from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


# def print_prompt(prompt):
#     print('='*20)
#     print(prompt.to_string())
#     print('='*20)
#     return prompt

class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()


    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def ensure_knowledge_base(self) -> int:
        """启动时检查向量库；空库自动重建，有数据则同步变更。"""
        return self.vector_store.ensure_knowledge_base()

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        """只负责检索，返回原始 Document，供评测和上层回答复用。"""
        if k is None:
            return self.retriever.invoke(query)
        return self.vector_store.get_retriever(k=k).invoke(query)

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
        return self.chain.invoke(
            {
                "input": query,
                "context": self.build_context(context_docs),
            }
        )

    def retriever_docs(self, query: str) -> list[Document]:
        """兼容旧调用名，等价于 retrieve(query)。"""
        return self.retrieve(query)

    def rag_summarize(self, query: str) -> str:
        """检索 + 生成的一条龙入口，供 Agent 工具继续使用。"""
        return self.answer(query, self.retrieve(query))

if __name__ == '__main__':
    rag = RagSummarizeService()
    rag.ensure_knowledge_base()
    print(rag.rag_summarize('小户型适合哪些扫地机器人'))