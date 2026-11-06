import csv
import json
import os
import random

from langchain_core.tools import tool

from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

rag = RagSummarizeService()

user_ids = ['1001', '1002', '1003', '1004', '1005', '1006', '1007', '1008', '1009', '1010']
month_arr = ['2026-06', '2026-07', '2026-08', '2026-09', '2026-10', '2026-11']

external_data: dict[str, dict[str, dict[str, str]]] = {}


def _clean_cell(value: str | None) -> str:
    """CSV 里用字面量 \\n 表示换行，转成真实换行，避免报告里出现反斜杠。"""
    return (value or "").replace("\\n", "\n").strip()


def ensure_rag_ready() -> int:
    """启动自检：空库自动灌库，有数据时同步新增/变更文件。"""
    return rag.ensure_knowledge_base()


@tool(description='从向量存储中检索参考资料')
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(description='获取城市天气，以信息字符串形式返回')
def get_weather(city: str) -> str:
    return f'城市{city}天气为晴天，气温为26摄氏度，空气湿度50%，南风1级'


@tool(description='获取用户所在城市的名称，以纯字符串形式返回')
def get_user_location() -> str:
    return random.choice(['厦门', '南京', '杭州'])


@tool(description='获取用户的ID，以纯字符串形式返回')
def get_user_id() -> str:
    return random.choice(user_ids)


@tool(description='获取当前月份，以纯字符串形式返回')
def get_current_month() -> str:
    return random.choice(month_arr)


def generate_external_data() -> None:
    if external_data:
        return

    external_data_path = get_abs_path(agent_conf['external_data_path'])
    if not os.path.exists(external_data_path):
        raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

    with open(external_data_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        expected_columns = {'用户ID', '特征', '清洁效率', '耗材', '对比', '时间'}
        missing_columns = expected_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"外部数据缺少列：{sorted(missing_columns)}")

        for line_no, row in enumerate(reader, start=2):
            user_id = (row.get('用户ID') or '').strip()
            month = (row.get('时间') or '').strip()
            if not user_id or not month:
                logger.warning(f'[外部数据] 第{line_no}行缺少用户ID或月份，跳过')
                continue

            external_data.setdefault(user_id, {})[month] = {
                '特征': _clean_cell(row.get('特征')),
                '效率': _clean_cell(row.get('清洁效率')),
                '耗材': _clean_cell(row.get('耗材')),
                '对比': _clean_cell(row.get('对比')),
            }


@tool(description='从外部系统中获取用户的使用记录，返回 JSON 字符串；未使用则返回空字符串')
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()

    record = external_data.get(user_id, {}).get(month)
    if not record:
        logger.warning(f'[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录')
        return ""

    return json.dumps(record, ensure_ascii=False)


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"


# if __name__ == '__main__':
#     print(fetch_external_data('1006', '2026-07'))
