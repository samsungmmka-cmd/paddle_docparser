"""HTTP API.  Run:  uvicorn docparse.api:app --host 0.0.0.0 --port 8080

Every /v1 request must send the key from DOCPARSE_API_KEY in the X-API-Key header.
The service refuses to start without a key.
"""
import hmac
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse

from .loaders import InvalidDocumentError, InvalidPageRangeError, TooManyPagesError, UnsupportedFileError
from .parser import DocumentParser
from .utils import to_markdown

MIN_KEY_LENGTH = 16


def create_app(parser: DocumentParser | None = None, warmup: bool = True) -> FastAPI:
    state = {"parser": parser, "ready": False}

    @asynccontextmanager
    async def lifespan(_app):
        if state["parser"] is None:
            state["parser"] = DocumentParser()
        if len(state["parser"].s.api_key) < MIN_KEY_LENGTH:
            raise RuntimeError(
                f"DOCPARSE_API_KEY must be set to a secret of at least {MIN_KEY_LENGTH} characters. "
                "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\"")
        if warmup and hasattr(state["parser"].models, "warmup"):
            await run_in_threadpool(state["parser"].models.warmup)  # load models before traffic
        state["ready"] = True
        yield

    def require_api_key(x_api_key: str | None = Header(None)):
        expected = state["parser"].s.api_key.encode()
        # compare_digest: constant-time, so the key can't be guessed by timing responses
        if not x_api_key or not hmac.compare_digest(x_api_key.encode(), expected):
            raise HTTPException(401, "Missing or invalid API key.", headers={"WWW-Authenticate": "ApiKey"})

    app = FastAPI(title="PaddleOCR Document Parser", version="1.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "ready": state["ready"]}

    @app.post("/v1/parse", dependencies=[Depends(require_api_key)])
    async def parse(
        file: UploadFile = File(...),
        pages: str | None = Query(None, max_length=200, description="Pages to read, e.g. '1-3,5'"),
        output: str = Query("json", pattern="^(json|markdown)$"),
        include_lines: bool = Query(False, description="Include every text line with its box"),
    ):
        p: DocumentParser = state["parser"]
        data = await file.read()
        if len(data) > p.s.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"File is larger than {p.s.max_upload_mb} MB.")
        if not data:
            raise HTTPException(400, "Empty file.")
        try:
            result = await run_in_threadpool(p.parse, data, file.filename, pages)
        except UnsupportedFileError as e:
            raise HTTPException(415, str(e))
        except TooManyPagesError as e:
            raise HTTPException(413, str(e))
        except (InvalidPageRangeError, InvalidDocumentError) as e:
            raise HTTPException(400, str(e))

        if output == "markdown":
            return PlainTextResponse(to_markdown(result), media_type="text/markdown")
        if not include_lines:
            for page in result["pages"]:
                for block in page["blocks"]:
                    block.pop("lines", None)
        return result

    return app


app = create_app()
