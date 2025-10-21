from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from google_play import create_managed_inapp, list_inapp_products

load_dotenv()

app = FastAPI(title="iap-manager")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/", StaticFiles(directory="static", html=True), name="static")


class CreateInAppRequest(BaseModel):
    sku: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    price_won: int = Field(..., ge=100)


@app.get("/api/inapp/list")
async def api_list_inapp(token: str | None = Query(default=None)):
    try:
        result = list_inapp_products(page_token=token)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/inapp/create")
async def api_create_inapp(payload: CreateInAppRequest):
    try:
        created = create_managed_inapp(
            sku=payload.sku,
            title=payload.title,
            description=payload.description,
            price_won=payload.price_won,
        )
        return created
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
async def health_check():
    return {"status": "ok"}
