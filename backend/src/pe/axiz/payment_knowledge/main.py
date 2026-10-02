from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pe.axiz.payment_knowledge.api.routes import router
from pe.axiz.payment_knowledge.config import get_settings
from pe.axiz.payment_knowledge.container import get_container

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    container = get_container()
    try:
        yield
    finally:
        await container.close()


app = FastAPI(title=settings.app_name, version="0.1.5", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.parsed_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


def run() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
