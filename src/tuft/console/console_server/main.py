import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router as api_router


app = FastAPI(title="TuFT Console Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("TUFT_GUI_URL", "http://localhost:10613/")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/")
def read_root():
    return {"message": "TuFT Console Server is running."}
