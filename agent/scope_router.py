"""服务范围的纯函数路由：报告 / 越界 / 普通问答。

越界判断只覆盖「明确不属于扫地机器人客服」的请求类别，命中后直接返回
固定拒答，不再让模型尝试。规则可测、可审计，避免模型顺手把代码、翻译、
医疗、投资等任务也做了。
"""

from __future__ import annotations

import re

from agent.report_router import is_report_query

OUT_OF_SCOPE_ANSWER = (
    "抱歉，这个请求超出了我的服务范围。我只负责扫地/扫拖机器人的选购、使用、"
    "维护和故障排查；编程、翻译、医疗、投资、订票等通用服务我无法处理。"
    "如果是其他家电的问题，建议查看对应品牌的说明书或联系其售后。"
)

# 每一条都要求「动作 + 对象」同时出现，避免把知识问答误伤。
# 例如「拖布能用洗衣机洗吗」是正常问题，不能因为出现「洗衣机」就拒答。
_OUT_OF_SCOPE_PATTERNS = (
    # 代写代码 / 脚本 / 爬虫
    re.compile(
        r"(帮我|请|给我|替我).{0,6}(写|生成|实现|编|做).{0,12}"
        r"(python|java|javascript|js|c\+\+|代码|脚本|爬虫|程序)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(写|生成|实现).{0,8}(python|java|javascript|js|c\+\+|爬虫|代码)",
        re.IGNORECASE,
    ),
    # 通用翻译
    re.compile(r"(翻译|译成|translate)", re.IGNORECASE),
    # 作文 / 诗 / 小说 / 文案 / 邮件
    re.compile(r"(写|作|生成).{0,12}(作文|诗|小说|故事|文案|邮件|情书)"),
    # 医疗与用药：既要症状，也要「求医问药」的动作，避免误伤「家里有人感冒怎么用」。
    re.compile(
        r"(吃什么药|用药|病症)|(我|孩子|宝宝|老人|家里人).{0,6}(感冒|发烧|头疼|咳嗽)"
    ),
    # 投资建议
    re.compile(r"(股票|基金|彩票|理财|投资建议|涨停)"),
    # 票务 / 外卖 / 充值
    re.compile(r"(订|买|退).{0,10}(机票|火车票|酒店|外卖)"),
    re.compile(r"(外卖|充话费|话费充值)"),
    # 影音娱乐推荐
    re.compile(r"(电影|电视剧|综艺|动漫).{0,8}(推荐|好看|看什么)"),
    re.compile(r"推荐.{0,12}(电影|电视剧|综艺|动漫)"),
    # 其他家电的维修 / 故障（不拦「用洗衣机洗拖布」这类正常用法）
    re.compile(r"(上门|维修|修理).{0,8}(洗衣机|冰箱|空调|电视|手机)"),
    re.compile(r"(洗衣机|冰箱|空调|电视).{0,6}(报错|报\s*E\d+|故障|售后)"),
    # 手机只拦明确的手机故障，不拦「扫地机连不上手机 App」。
    re.compile(r"手机.{0,6}(坏了|黑屏|开不了机|售后|维修)"),
    # 扫地机器人不存在的功能
    re.compile(r"(洗碗|做饭|炒菜|遛狗|看孩子|洗衣服|擦玻璃|拖窗户)"),
    re.compile(r"(当|当作|作为|用来).{0,4}(监控|摄像头)"),
)


def is_out_of_scope_query(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    return any(pattern.search(value) for pattern in _OUT_OF_SCOPE_PATTERNS)


def route_intent(text: str) -> str:
    """返回 report / out_of_scope / qa；报告优先，避免「写使用报告」被判越界。"""
    if is_report_query(text):
        return "report"
    if is_out_of_scope_query(text):
        return "out_of_scope"
    return "qa"


__all__ = [
    "OUT_OF_SCOPE_ANSWER",
    "is_out_of_scope_query",
    "route_intent",
]
