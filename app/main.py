import logging
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from app.config import get_settings
from app.errors import AppError
from app.schemas import ErrorBody, ErrorResponse, LinkAnalysisRequest, LinkAnalysisResponse
from app.services.analysis_service import AnalysisService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

app = FastAPI(
    title="AI Website Link Analysis API",
    version="1.0.0",
    description=(
        "Analyses a website with an LLM and returns competitor, search-query, SEO, services "
        "and target-audience insights."
    ),
)


@lru_cache
def get_analysis_service() -> AnalysisService:
    return AnalysisService(get_settings())


def _error(status: int, code: str, message: str, details=None) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message, details=details))
    return JSONResponse(status_code=status, content=body.model_dump())


@app.exception_handler(AppError)
async def handle_app_error(_: Request, exc: AppError):
    return _error(exc.status_code, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_: Request, exc: RequestValidationError):
    details = [{"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]} for e in exc.errors()]
    return _error(422, "invalid_request", 'Send a JSON body like {"url": "https://example.com"}.', details)


@app.exception_handler(Exception)
async def handle_unexpected(_: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return _error(500, "internal_error", "An unexpected error occurred.")


STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
async def ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok"}


@app.post(
    "/link-analysis",
    response_model=LinkAnalysisResponse,
    tags=["analysis"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid or blocked URL"},
        422: {"model": ErrorResponse, "description": "Bad request body or website unreachable/unreadable"},
        502: {"model": ErrorResponse, "description": "LLM provider failure or invalid LLM output"},
        503: {"model": ErrorResponse, "description": "LLM API key not configured"},
    },
)
async def link_analysis(
    request: LinkAnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service),
):
    return await service.analyze(request.url)
