import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from dotenv import load_dotenv

from google_play import create_managed_inapp, list_inapp_products

load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(title="iap-manager")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_files = StaticFiles(directory="static", html=True)

app.mount("/static", static_files, name="static")


@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


class Translation(BaseModel):
    language: str = Field(..., min_length=2)
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)


class CreateInAppRequest(BaseModel):
    sku: str = Field(..., min_length=1)
    default_language: str = Field("ko-KR", min_length=2)
    price_won: int | None = Field(default=None, ge=100)
    pricing_template_id: str | None = Field(default=None, min_length=1)
    translations: list[Translation] = Field(default_factory=list)

    @model_validator(mode="after")
    def ensure_pricing(cls, model: "CreateInAppRequest"):
        price = model.price_won
        template_id = model.pricing_template_id
        if price is None and not template_id:
            raise ValueError("가격 또는 가격 템플릿 중 하나는 반드시 입력해야 합니다.")
        if price is not None and template_id:
            raise ValueError("가격과 가격 템플릿을 동시에 설정할 수 없습니다.")
        translations = model.translations or []
        languages = {t.language for t in translations}
        default_language = model.default_language
        if default_language not in languages:
            raise ValueError("기본 언어에 대한 번역 정보를 입력해야 합니다.")
        return model

    @property
    def default_translation(self) -> Translation:
        for translation in self.translations:
            if translation.language == self.default_language:
                return translation
        raise ValueError("기본 언어 번역을 찾을 수 없습니다.")


@app.get("/api/inapp/list")
async def api_list_inapp(token: str | None = Query(default=None)):
    try:
        result = list_inapp_products(page_token=token)
        return result
    except Exception as exc:
        logger.exception("Failed to list in-app products")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/inapp/create")
async def api_create_inapp(payload: CreateInAppRequest):
    try:
        created = create_managed_inapp(
            sku=payload.sku,
            default_language=payload.default_language,
            default_title=payload.default_translation.title,
            default_description=payload.default_translation.description,
            price_won=payload.price_won,
            pricing_template_id=payload.pricing_template_id,
            translations=[t.dict() for t in payload.translations],
        )
        return created
    except Exception as exc:
        logger.exception("Failed to create managed in-app product")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
async def health_check():
    return {"status": "ok"}
