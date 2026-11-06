import os,hashlib
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader,TextLoader


def get_file_md5_hex(filepath: str) -> str | None:  # 获取文件的 md5 十六进制字符串
    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件{filepath}不存在")
        return None

    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]路径{filepath}不是文件")
        return None

    md5_obj = hashlib.md5()
    chunk_size = 4096
    try:
        with open(filepath, 'rb') as f:#必须二进制读取
            while chunk := f.read(chunk_size) :
                md5_obj.update(chunk)
            md5_hex=md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"计算文件{filepath}md5出错，{str(e)}")
        return None


def listdir_with_allowed_type(
    path: str,
    allowed_types: tuple[str, ...],
) -> list[str]:  # 返回文件夹内的文件列表（允许的文件后缀）
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return []

    return [
        os.path.join(path, filename)
        for filename in sorted(os.listdir(path))
        if filename.endswith(allowed_types)
    ]


def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    return PyPDFLoader(filepath, passwd).load()

def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath,encoding="utf-8").load()