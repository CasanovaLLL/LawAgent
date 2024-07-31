from datetime import datetime

__all__ = [
    'generate_timestamp'
]


def generate_timestamp():
    """
    生成当前时间的yyyyMMddhhmmss格式的时间戳字符串。
    """
    now = datetime.now()  # 获取当前时间
    formatted_time = now.strftime('%Y%m%d%H%M%S')  # 格式化时间字符串
    return formatted_time
