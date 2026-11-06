import hashlib
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import embed_model
from utils.config_handler import chroma_conf
from utils.file_handler import (
    get_file_md5_hex,
    listdir_with_allowed_type,
    pdf_loader,
    txt_loader,
)
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class VectorStoreService:
    def __init__(self):
        persist_directory = chroma_conf["persist_directory"]
        # Chroma 的相对路径按进程工作目录解析；这里统一锚到项目根。
        if not os.path.isabs(persist_directory):
            persist_directory = get_abs_path(persist_directory)
        self.persist_directory = persist_directory

        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=self.persist_directory,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self, k: int | None = None):
        """返回检索器；k 为空时使用配置默认值，便于评测时临时调整召回数。"""
        return self.vector_store.as_retriever(
            search_kwargs={"k": k if k is not None else chroma_conf["k"]}
        )

    def _documents_for(self, read_path: str) -> list[Document]:
        suffix = os.path.splitext(read_path)[1].lower()
        if suffix == ".txt":
            return txt_loader(read_path)
        if suffix == ".pdf":
            return pdf_loader(read_path)
        return []

    def _indexed_md5(self, source: str) -> str | None:
        """从 Chroma 元数据里读某个文件最后一次入库时的 md5。"""
        result = self.vector_store.get(
            where={"source": source},
            include=["metadatas"],
            limit=1,
        )
        metadatas = result.get("metadatas") or []
        if not metadatas:
            return None
        return (metadatas[0] or {}).get("file_md5")

    def _chunk_ids(self, source: str, md5_hex: str, count: int) -> list[str]:
        # 同一文件内容不变时 ID 稳定；不同文件即使内容相同也不会撞 ID。
        source_key = hashlib.md5(source.encode("utf-8")).hexdigest()
        return [f"{source_key}:{md5_hex}:{index}" for index in range(count)]

    def _prune_deleted_sources(self, current_sources: set[str]) -> None:
        """删除 data 目录里已经不存在的文件对应的旧向量。"""
        result = self.vector_store.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        indexed_sources = {
            metadata["source"]
            for metadata in metadatas
            if metadata and metadata.get("source")
        }

        for stale_source in sorted(indexed_sources - current_sources):
            logger.info(f"[加载知识库] {stale_source} 已不在 data 目录，删除旧向量")
            self.vector_store.delete(where={"source": stale_source})

    def _count(self) -> int:
        return self.vector_store._collection.count()

    def load_document(self) -> int:
        """把 data 目录里的文件同步进向量库，返回本次新增的 chunk 数。

        以 Chroma 中每个 chunk 的 file_md5 元数据作为去重依据：
        - 文件没变：跳过；
        - 文件新增或内容变化：先按 source 删旧 chunk，再写入新 chunk；
        - data 里删掉的文件：对应向量一并清理。
        这样即使向量库被清空，也会自动重建，不再被外部 md5 台账卡住。
        """
        allowed_files_path = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )
        if not allowed_files_path:
            logger.warning(f"[加载知识库] {chroma_conf['data_path']} 下没有可加载的文件")
            return 0

        current_sources: set[str] = set()
        added_count = 0

        for path in allowed_files_path:
            current_sources.add(path)
            md5_hex = get_file_md5_hex(path)
            if not md5_hex:
                logger.error(f"[加载知识库] 无法计算 {path} 的 md5，跳过")
                continue

            indexed_md5 = self._indexed_md5(path)
            if indexed_md5 == md5_hex:
                logger.info(f"[加载知识库] {path} 已是最新版本，跳过")
                continue

            try:
                documents = self._documents_for(path)
                if not documents:
                    logger.warning(f"[加载知识库] {path} 内无有效文本内容，跳过")
                    continue

                split_documents = self.splitter.split_documents(documents)
                if not split_documents:
                    logger.warning(f"[加载知识库] {path} 分片后无有效文本内容，跳过")
                    continue

                for document in split_documents:
                    document.metadata["source"] = path
                    document.metadata["file_md5"] = md5_hex

                ids = self._chunk_ids(path, md5_hex, len(split_documents))
                # 先删旧版本，避免文件内容变化后新旧 chunk 同时留在库里。
                self.vector_store.delete(where={"source": path})
                self.vector_store.add_documents(split_documents, ids=ids)

                added_count += len(split_documents)
                logger.info(
                    f"[加载知识库] {path} 同步成功，写入 {len(split_documents)} 个 chunk"
                )
            except Exception as e:
                logger.error(
                    f"[加载知识库] {path} 添加失败，错误信息为：{e}",
                    exc_info=True,
                )
                continue

        self._prune_deleted_sources(current_sources)
        return added_count

    def ensure_knowledge_base(self) -> int:
        """启动自检：库为空时自动重建；有数据时顺带同步新增/变更文件。"""
        count_before = self._count()
        logger.info(f"[加载知识库] 当前向量数：{count_before}")

        if count_before == 0:
            logger.warning("[加载知识库] 向量库为空，开始自动重建")

        self.load_document()
        count_after = self._count()
        logger.info(f"[加载知识库] 同步完成，当前向量数：{count_after}")

        if count_after == 0:
            raise RuntimeError(
                "知识库为空：data 目录没有有效文件，或 embedding 调用失败，请查看 log/。"
            )
        return count_after


if __name__ == "__main__":
    vs = VectorStoreService()
    vs.ensure_knowledge_base()
    retriever = vs.get_retriever()
    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-" * 20)
