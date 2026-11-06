from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="LLMProxy",
    description="Minimalist AI Router & Proxy with Hot-Reloadable Plugin Architecture",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for the Black and White dashboard
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
index_file = static_dir / "index.html"
if not index_file.exists():
    index_file.write_text("<!DOCTYPE html><html><head><title>LLMProxy</title></head><body><h1>LLMProxy</h1></body></html>")

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/health")
async def health_check():
    return {"status": "ok", "app": "llmproxy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("llmproxy.main:app", host="0.0.0.0", port=8000, reload=True)
