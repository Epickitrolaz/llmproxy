import logging
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from llmproxy.core.manager import plugin_manager
from llmproxy.core.security import cors_allow_credentials, cors_origins

logger = logging.getLogger(__name__)


def sanitize_ssl_env() -> None:
    """Clear SSL certificate environment variables if they point to non-existent files to prevent FileNotFoundError."""
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        val = os.environ.get(var)
        if val and not os.path.exists(val):
            logger.warning(
                "Clearing invalid %s='%s' (file not found) to prevent SSL failures",
                var,
                val,
            )
            del os.environ[var]
    for var in ("SSL_CERT_DIR",):
        val = os.environ.get(var)
        if val and not os.path.isdir(val):
            logger.warning(
                "Clearing invalid %s='%s' (directory not found) to prevent SSL failures",
                var,
                val,
            )
            del os.environ[var]


sanitize_ssl_env()

app = FastAPI(
    title="LLMProxy",
    description="Minimalist AI Router & Proxy with Hot-Reloadable Plugin Architecture",
    version="0.1.0",
)

_cors_origins = cors_origins()
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=cors_allow_credentials(_cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Startup & shutdown events
@app.on_event("startup")
async def startup_event():
    plugin_manager.discover_and_load_all()

@app.on_event("shutdown")
async def shutdown_event():
    plugin_manager.flush()

# Try to register API routers if available
try:
    from llmproxy.api.router import router as api_v1_router
    app.include_router(api_v1_router)
except ImportError:
    pass

try:
    from llmproxy.dashboard.auth import router as dashboard_auth_router
    app.include_router(dashboard_auth_router)
except ImportError:
    pass

try:
    from llmproxy.dashboard.router import router as dashboard_api_router
    app.include_router(dashboard_api_router)
except ImportError:
    pass

try:
    from llmproxy.api.management import router as management_router
    app.include_router(management_router)
except ImportError:
    pass

# Static files for the Black and White dashboard
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
index_file = static_dir / "index.html"
if not index_file.exists():
    index_file.write_text("<!DOCTYPE html><html><head><title>LLMProxy</title></head><body><h1>LLMProxy</h1></body></html>")

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def root_index():
    return FileResponse(str(index_file))

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "app": "llmproxy",
        "plugins_count": len(plugin_manager.plugins),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("llmproxy.main:app", host="0.0.0.0", port=8000, reload=True)
