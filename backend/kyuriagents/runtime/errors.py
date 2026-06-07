"""User-facing runtime error normalization."""


_CONTACT = "210825684@qq.com"
_QUOTA_MARKERS = (
    "insufficient_quota",
    "quota",
    "balance",
    "billing",
    "credit",
    "rate_limit",
    "rate limit",
    "too many requests",
    "throttl",
    "429",
    "402",
)
_AUTH_MARKERS = (
    "invalid api key",
    "incorrect api key",
    "unauthorized",
    "401",
    "403",
    "dashscope_api_key",
    "api-key",
    "apikey",
)
_NETWORK_MARKERS = (
    "connection error",
    "connecterror",
    "read timed out",
    "timeout",
    "temporarily unavailable",
    "service unavailable",
    "502",
    "503",
    "504",
)


def public_error_message(error: object) -> str:
    """Return a safe, helpful message for runtime failures.

    Args:
        error: Exception or error text raised by an LLM, embedding, or service call.

    Returns:
        Chinese user-facing error message without provider stack traces.
    """
    text = str(error)
    lowered = text.lower()
    if any(marker in lowered for marker in _QUOTA_MARKERS):
        return f"模型服务暂时不可用: 当前项目的模型额度可能已用尽或请求过于频繁。请稍后再试, 或联系作者 {_CONTACT}。"
    if any(marker in lowered for marker in _AUTH_MARKERS):
        return f"模型服务配置异常: API Key 可能无效或未配置。请联系作者 {_CONTACT}。"
    if any(marker in lowered for marker in _NETWORK_MARKERS):
        return f"模型服务暂时连接失败, 可能是网络波动或上游服务繁忙。请稍后再试; 如果持续出现, 请联系作者 {_CONTACT}。"
    if "dashscope_api_key" in lowered or "dashscope api key" in lowered:
        return f"模型服务尚未配置 API Key。请联系作者 {_CONTACT}。"
    return text or f"请求失败。请稍后再试, 或联系作者 {_CONTACT}。"


__all__ = ["public_error_message"]
