from functools import lru_cache

from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


def _read_prompt(config_key: str) -> str:
    """按 prompts.yml 中的配置项读取提示词文件。"""
    try:
        prompt_path = get_abs_path(prompts_conf[config_key])
    except KeyError:
        logger.error(f"[加载提示词] prompts.yml 缺少配置项 {config_key}")
        raise

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"[加载提示词] 读取 {prompt_path} 出错：{str(e)}")
        raise


# 提示词是只读常量，进程内缓存，避免每次请求都读盘；
# 修改提示词文件后需要重启应用（或调用 load_*_prompts.cache_clear()）。
@lru_cache(maxsize=None)
def load_system_prompts() -> str:
    return _read_prompt("main_prompt_path")


@lru_cache(maxsize=None)
def load_rag_prompts() -> str:
    return _read_prompt("rag_summarize_prompt_path")


@lru_cache(maxsize=None)
def load_report_prompts() -> str:
    return _read_prompt("report_prompt_path")


if __name__ == '__main__':
    print(load_system_prompts())
