# -*- coding: utf-8 -*-
import sys
import json
import logging
import time
from typing import Dict, Any
import requests
from config import config

logging.basicConfig(level=logging.INFO, filename='output.log', datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')

# 复制原有的 callHandlers
origin_callHandlers = logging.Logger.callHandlers
# 屏幕重定向handler
stdout_handler = logging.StreamHandler(sys.stdout)


# call_handlers 重新定义
def call_handlers(self, record):
    try:
        return origin_callHandlers(self, record)
    finally:
        # 这里上报日志系统es；大于等于ERROR等级的log才输出
        if record.levelno >= logging.INFO and config.REMOTE_WRITE:
            es_logger(record.message, record.levelno)


# es_logger 添加 es 日志上报, 实现http 接口
def es_logger(message: str, level: int) -> str:
    now = int(round(time.time() * 1000))
    level_name = logging.getLevelName(level)
    id = str(hash(message))
    param = {
        "logId": id,
        "appId": "app30002",
        "requestId": id,
        "logLevel": level_name,
        "code": 0,
        "message": message,
        "source": "DOSSEN-RMS-ALG_API",
        "createTime": now
    }
    res = "fail"
    try:
        res = post(param, config.LOG_URL)
        if res != "ok":
            print("res", res)
    except Exception as e:
        print("exception", e)
    return res


# post 发送http post 请求.
def post(param: Dict[str, Any], url: str) -> str:
    """
    :param param: 传入参数
    :param url: 请求url
    :return: 返回str类型
    """
    response = requests.post(url, data=json.dumps(param), headers={'Content-Type': 'application/json'})
    #  成功返回ok
    return response.json()["message"]


# 替换原来的callHandlers方法
logging.Logger.callHandlers = call_handlers
logging.getLogger().addHandler(stdout_handler)

# 外部调用logger
logger = logging.getLogger(__name__)

#  执行python 测试代码
if __name__ == '__main__':
    logger.info("...... error for test log.....")
