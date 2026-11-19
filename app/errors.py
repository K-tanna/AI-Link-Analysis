"""Application-level errors. Each maps to an HTTP status and a stable error code."""


class AppError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidURLError(AppError):
    status_code = 400
    code = "invalid_url"


class BlockedURLError(AppError):
    """URL is syntactically valid but points somewhere we refuse to fetch (SSRF protection)."""

    status_code = 400
    code = "blocked_url"


class WebsiteFetchError(AppError):
    status_code = 422
    code = "website_unreachable"


class EmptyContentError(AppError):
    status_code = 422
    code = "no_extractable_content"


class LLMNotConfiguredError(AppError):
    status_code = 503
    code = "llm_not_configured"


class LLMError(AppError):
    status_code = 502
    code = "llm_unavailable"


class LLMOutputError(AppError):
    status_code = 502
    code = "llm_invalid_output"
