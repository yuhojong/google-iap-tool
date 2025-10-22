import csv
import io
import logging
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Literal, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from dotenv import load_dotenv

from google_play import (
    create_managed_inapp,
    delete_inapp_product,
    get_all_inapp_products,
    list_inapp_products,
    update_managed_inapp,
)
from price_templates import (
    PriceTemplate,
    generate_price_templates_from_products,
    get_template_by_id,
    index_templates_by_price_micros,
)

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
    price_template_id: str | None = Field(default=None, min_length=1)
    translations: list[Translation] = Field(default_factory=list)

    @model_validator(mode="after")
    def ensure_pricing(cls, model: "CreateInAppRequest"):
        translations = model.translations or []
        languages = {t.language for t in translations}
        default_language = model.default_language
        if default_language not in languages:
            raise ValueError("기본 언어에 대한 번역 정보를 입력해야 합니다.")

        has_manual_price = model.price_won is not None
        has_template = model.price_template_id is not None
        if has_manual_price == has_template:
            raise ValueError("가격 템플릿 또는 직접 입력 가격 중 하나만 선택해야 합니다.")
        return model


class ListingPayload(BaseModel):
    title: str
    description: str


class PricePayload(BaseModel):
    priceMicros: str
    currency: str


class ImportProductPayload(BaseModel):
    sku: str
    status: str = Field(default="active")
    default_language: str
    default_price: PricePayload
    listings: Dict[str, ListingPayload]
    prices: Optional[Dict[str, Any]] = None

    @field_validator("listings")
    @classmethod
    def ensure_default_language_listing(
        cls, listings: Dict[str, ListingPayload], info: Dict[str, Any]
    ) -> Dict[str, ListingPayload]:
        default_language = info.data.get("default_language")
        if default_language and default_language not in listings:
            raise ValueError("기본 언어 번역 정보가 필요합니다.")
        return listings


class UpdateInAppRequest(BaseModel):
    default_language: str = Field(..., min_length=2)
    status: str = Field(default="active", min_length=1)
    default_price: PricePayload
    listings: Dict[str, ListingPayload]
    prices: Optional[Dict[str, PricePayload]] = None

    @field_validator("listings")
    @classmethod
    def ensure_default_language_listing(
        cls, listings: Dict[str, ListingPayload], info: Dict[str, Any]
    ) -> Dict[str, ListingPayload]:
        default_language = info.data.get("default_language")
        if default_language and default_language not in listings:
            raise ValueError("기본 언어 번역 정보가 필요합니다.")
        return listings


class ImportOperation(BaseModel):
    action: Literal["create", "update", "delete"]
    sku: str
    data: Optional[ImportProductPayload] = None


class ImportApplyRequest(BaseModel):
    operations: List[ImportOperation]


def _collect_languages_from_products(products: Iterable[Dict[str, Any]]) -> List[str]:
    languages: set[str] = set()
    for item in products:
        listings = item.get("listings") or {}
        for language in listings.keys():
            if isinstance(language, str):
                languages.add(language)
    return sorted(languages)


def _canonicalize_product(item: Dict[str, Any]) -> Dict[str, Any]:
    listings = item.get("listings") or {}
    normalized_listings: Dict[str, Dict[str, str]] = {}
    for language, listing in listings.items():
        if not isinstance(language, str) or not isinstance(listing, dict):
            continue
        title = listing.get("title")
        description = listing.get("description")
        if title is None and description is None:
            continue
        normalized_listings[language] = {
            "title": title or "",
            "description": description or "",
        }

    default_price = item.get("defaultPrice") or {}
    prices = item.get("prices") or {}
    normalized_prices: Dict[str, Dict[str, str]] = {}
    for region, price in prices.items():
        if not isinstance(region, str) or not isinstance(price, dict):
            continue
        price_micros = price.get("priceMicros")
        currency = price.get("currency")
        if price_micros is None and currency is None:
            continue
        normalized_prices[region] = {
            "priceMicros": str(price_micros or ""),
            "currency": currency or "",
        }

    return {
        "sku": item.get("sku", ""),
        "status": item.get("status", ""),
        "default_language": item.get("defaultLanguage", ""),
        "default_price": {
            "priceMicros": str(default_price.get("priceMicros") or ""),
            "currency": default_price.get("currency") or "",
        },
        "listings": normalized_listings,
        "prices": normalized_prices or None,
    }


def _normalize_price_won(value: str) -> tuple[str, str]:
    stripped = value.strip().replace(",", "").replace("_", "")
    if not stripped:
        raise ValueError("KRW 가격 값을 입력해주세요.")
    try:
        decimal_value = Decimal(stripped)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("KRW 가격은 숫자여야 합니다.") from exc
    if decimal_value <= 0:
        raise ValueError("KRW 가격은 0보다 커야 합니다.")
    scaled = decimal_value * Decimal("1000000")
    if scaled != scaled.to_integral_value():
        raise ValueError("KRW 가격은 소수점 여섯째 자리까지만 입력할 수 있습니다.")
    price_micros = scaled.to_integral_value()
    normalized_price = decimal_value.normalize()
    price_text = format(normalized_price, "f")
    return str(int(price_micros)), price_text


def _format_price_won_from_micros(price_micros: str, currency: str) -> str:
    if currency.upper() != "KRW":  # type: ignore[attr-defined]
        return ""
    try:
        decimal_value = Decimal(price_micros) / Decimal("1000000")
    except (InvalidOperation, ValueError):
        return ""
    text = f"{decimal_value:,.6f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _compute_changes(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    changes: Dict[str, Any] = {}
    if before.get("status") != after.get("status"):
        changes["status"] = {"from": before.get("status"), "to": after.get("status")}
    if before.get("default_language") != after.get("default_language"):
        changes["default_language"] = {
            "from": before.get("default_language"),
            "to": after.get("default_language"),
        }

    before_price = before.get("default_price") or {}
    after_price = after.get("default_price") or {}
    if (
        before_price.get("priceMicros") != after_price.get("priceMicros")
        or before_price.get("currency") != after_price.get("currency")
    ):
        changes["default_price"] = {
            "from": before_price,
            "to": after_price,
        }

    listing_changes: Dict[str, Any] = {}
    before_listings = before.get("listings") or {}
    after_listings = after.get("listings") or {}
    languages = set(before_listings.keys()) | set(after_listings.keys())
    for language in sorted(languages):
        prev = before_listings.get(language)
        nxt = after_listings.get(language)
        if prev != nxt:
            listing_changes[language] = {"from": prev, "to": nxt}
    if listing_changes:
        changes["listings"] = listing_changes

    before_prices = before.get("prices") or {}
    after_prices = after.get("prices") or {}
    if before_prices != after_prices:
        changes["prices"] = {"from": before_prices, "to": after_prices}
    return changes


def _parse_import_csv(content: bytes) -> tuple[List[Dict[str, Any]], List[str]]:
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="CSV 파일은 UTF-8 인코딩이어야 합니다.") from exc

    reader = csv.DictReader(io.StringIO(decoded))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV 헤더를 찾을 수 없습니다.")

    languages = sorted({name.split("title_", 1)[1] for name in reader.fieldnames if name.startswith("title_")})
    rows: List[Dict[str, Any]] = []
    seen_skus: set[str] = set()

    for index, row in enumerate(reader, start=2):
        sku = (row.get("sku") or "").strip()
        if not sku:
            continue
        if sku in seen_skus:
            raise HTTPException(status_code=400, detail=f"CSV에 중복된 SKU가 있습니다: {sku}")
        seen_skus.add(sku)

        entry: Dict[str, Any] = {
            "row": index,
            "sku": sku,
            "status": (row.get("status") or "").strip() or None,
            "default_language": (row.get("default_language") or "").strip() or None,
            "price_won": (row.get("price_won") or "").strip() or None,
            "listings": {},
            "prices": None,
        }
        for language in languages:
            title_key = f"title_{language}"
            description_key = f"description_{language}"
            title = (row.get(title_key) or "").strip()
            description = (row.get(description_key) or "").strip()
            if title or description:
                entry["listings"][language] = {
                    "title": title,
                    "description": description,
                }

        rows.append(entry)

    return rows, languages


def _build_import_operations(
    rows: List[Dict[str, Any]],
    existing_products: Dict[str, Dict[str, Any]],
    templates_by_price_micros: Dict[str, PriceTemplate] | None = None,
) -> Dict[str, Any]:
    templates_by_price_micros = templates_by_price_micros or {}
    operations: List[Dict[str, Any]] = []
    summary = {"create": 0, "update": 0, "delete": 0}

    remaining_products = {sku: data for sku, data in existing_products.items()}

    for entry in rows:
        sku = entry["sku"]
        current = remaining_products.pop(sku, None)

        if current is None:
            default_language = entry["default_language"] or ""
            if not default_language:
                raise HTTPException(status_code=400, detail=f"{sku} 행: default_language 값을 입력해주세요.")

            price_won_value = entry["price_won"]
            if not price_won_value:
                raise HTTPException(status_code=400, detail=f"{sku} 행: price_won 값을 입력해주세요.")
            try:
                price_micros, _ = _normalize_price_won(price_won_value)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"{sku} 행: {exc}") from exc

            currency = "KRW"
            listings = entry["listings"]
            default_listing = listings.get(default_language)
            if not default_listing or not default_listing.get("title") or not default_listing.get("description"):
                raise HTTPException(status_code=400, detail=f"{sku} 행: 기본 언어 번역 제목과 설명이 필요합니다.")

            template = templates_by_price_micros.get(price_micros)
            regional_prices = None
            if template:
                pricing_payload = template.to_pricing_payload()
                regional_prices = pricing_payload.get("prices")

            new_product = {
                "sku": sku,
                "status": entry["status"] or "active",
                "default_language": default_language,
                "default_price": {
                    "priceMicros": price_micros,
                    "currency": currency,
                },
                "listings": listings,
                "prices": regional_prices,
            }
            operations.append({"action": "create", "sku": sku, "data": new_product})
            summary["create"] += 1
            continue

        default_language = entry["default_language"] or current.get("default_language") or ""
        if not default_language:
            raise HTTPException(status_code=400, detail=f"{sku} 행: 기본 언어 정보를 확인할 수 없습니다.")

        status = entry["status"] or current.get("status") or "active"

        applied_template = False
        template_prices: Optional[Dict[str, Any]] = None
        if entry["price_won"]:
            try:
                price_micros, _ = _normalize_price_won(entry["price_won"])
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"{sku} 행: {exc}") from exc
            currency = "KRW"
            template = templates_by_price_micros.get(price_micros)
            if template:
                pricing_payload = template.to_pricing_payload()
                template_prices = pricing_payload.get("prices")
                applied_template = True
        else:
            price_micros = current.get("default_price", {}).get("priceMicros")
            currency = current.get("default_price", {}).get("currency") or "KRW"

        if not price_micros or not currency:
            raise HTTPException(status_code=400, detail=f"{sku} 행: 기본 가격 정보를 확인할 수 없습니다.")

        new_listings = {lang: data.copy() for lang, data in (current.get("listings") or {}).items()}
        for language, listing in entry["listings"].items():
            if listing.get("title") and listing.get("description"):
                new_listings[language] = listing

        if default_language not in new_listings:
            raise HTTPException(status_code=400, detail=f"{sku} 행: 기본 언어 번역 정보를 제공해야 합니다.")

        if applied_template:
            new_prices = template_prices
        else:
            new_prices = {
                region: data.copy()
                for region, data in (current.get("prices") or {}).items()
                if isinstance(data, dict)
            }
            for region, price_payload in (entry.get("prices") or {}).items():
                new_prices[region] = price_payload.copy()
            if not new_prices:
                new_prices = None

        new_product = {
            "sku": sku,
            "status": status,
            "default_language": default_language,
            "default_price": {
                "priceMicros": price_micros,
                "currency": currency,
            },
            "listings": new_listings,
            "prices": new_prices,
        }

        changes = _compute_changes(current, new_product)
        if changes:
            operations.append(
                {
                    "action": "update",
                    "sku": sku,
                    "data": new_product,
                    "current": current,
                    "changes": changes,
                }
            )
            summary["update"] += 1

    for remaining in remaining_products.values():
        operations.append({"action": "delete", "sku": remaining.get("sku"), "current": remaining})
        summary["delete"] += 1

    return {"operations": operations, "summary": summary}

@app.get("/api/inapp/list")
async def api_list_inapp(token: str | None = Query(default=None)):
    try:
        result = list_inapp_products(page_token=token)
        return result
    except Exception as exc:
        logger.exception("Failed to list in-app products")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/inapp/export")
async def api_export_inapp() -> StreamingResponse:
    try:
        products = get_all_inapp_products()
        languages = _collect_languages_from_products(products)
        output = io.StringIO()
        fieldnames = ["sku", "status", "default_language", "price_won"]
        for language in languages:
            fieldnames.append(f"title_{language}")
            fieldnames.append(f"description_{language}")
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for item in products:
            canonical = _canonicalize_product(item)
            default_price = canonical.get("default_price") or {}
            default_price_micros = default_price.get("priceMicros") or ""
            default_currency = default_price.get("currency") or ""
            row = {
                "sku": canonical.get("sku", ""),
                "status": canonical.get("status", ""),
                "default_language": canonical.get("default_language", ""),
                "price_won": _format_price_won_from_micros(default_price_micros, default_currency)
                if default_price_micros
                else "",
            }
            listings = canonical.get("listings") or {}
            for language in languages:
                listing = listings.get(language) or {}
                row[f"title_{language}"] = listing.get("title", "")
                row[f"description_{language}"] = listing.get("description", "")
            writer.writerow(row)
        csv_content = output.getvalue()
        csv_bytes = csv_content.encode("utf-8-sig")
        headers = {"Content-Disposition": 'attachment; filename="iap-products.csv"'}
        return StreamingResponse(iter([csv_bytes]), media_type="text/csv", headers=headers)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to export in-app products")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/pricing/templates")
async def api_list_price_templates():
    try:
        products = get_all_inapp_products()
        templates = generate_price_templates_from_products(products)
        return {"templates": [template.to_response() for template in templates]}
    except Exception as exc:
        logger.exception("Failed to load price templates")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/inapp/create")
async def api_create_inapp(payload: CreateInAppRequest):
    try:
        regional_pricing = None
        if payload.price_template_id:
            products = get_all_inapp_products()
            templates = generate_price_templates_from_products(products)
            template = get_template_by_id(templates, payload.price_template_id)
            if not template:
                raise HTTPException(status_code=400, detail="유효하지 않은 가격 템플릿입니다.")
            regional_pricing = template.to_pricing_payload()
        created = create_managed_inapp(
            sku=payload.sku,
            default_language=payload.default_language,
            price_won=payload.price_won,
            regional_pricing=regional_pricing,
            translations=[t.dict() for t in payload.translations],
        )
        return created
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to create managed in-app product")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.put("/api/inapp/{sku}")
async def api_update_inapp(sku: str, payload: UpdateInAppRequest):
    try:
        listings_payload = {
            language: listing.model_dump()
            for language, listing in payload.listings.items()
        }
        prices_payload = (
            {region: price.model_dump() for region, price in payload.prices.items()}
            if payload.prices
            else None
        )
        update_managed_inapp(
            sku=sku,
            default_language=payload.default_language,
            status=payload.status,
            default_price=payload.default_price.model_dump(),
            listings=listings_payload,
            prices=prices_payload,
        )
        return {"status": "ok"}
    except HTTPException:
        raise
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc
    except Exception as exc:
        logger.exception("Failed to update managed in-app product")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/api/inapp/{sku}")
async def api_delete_inapp(sku: str):
    try:
        delete_inapp_product(sku=sku)
        return {"status": "ok"}
    except Exception as exc:
        logger.exception("Failed to delete in-app product")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/inapp/import/preview")
async def api_import_preview(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="CSV 파일이 비어 있습니다.")

    rows, csv_languages = _parse_import_csv(content)

    try:
        existing_products_raw = get_all_inapp_products()
        existing_products = {
            item.get("sku"): _canonicalize_product(item)
            for item in existing_products_raw
            if item.get("sku")
        }
        templates = generate_price_templates_from_products(existing_products_raw)
        templates_by_price = index_templates_by_price_micros(templates)
    except Exception as exc:
        logger.exception("Failed to load existing in-app products for import preview")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    operations_payload = _build_import_operations(rows, existing_products, templates_by_price)
    existing_languages = {
        lang
        for product in existing_products.values()
        for lang in (product.get("listings") or {}).keys()
    }
    languages = sorted(set(csv_languages) | existing_languages)

    return {
        "languages": languages,
        "operations": operations_payload["operations"],
        "summary": operations_payload["summary"],
    }


@app.post("/api/inapp/import/apply")
async def api_import_apply(request: ImportApplyRequest):
    results = {"create": 0, "update": 0, "delete": 0}

    try:
        for op in request.operations:
            if op.action == "create":
                if not op.data:
                    raise HTTPException(status_code=400, detail="create 작업에는 data가 필요합니다.")
                data = op.data
                translations = [
                    {"language": language, "title": listing.title, "description": listing.description}
                    for language, listing in data.listings.items()
                ]
                create_managed_inapp(
                    sku=data.sku,
                    default_language=data.default_language,
                    translations=translations,
                    default_price=data.default_price.model_dump(),
                    prices=data.prices,
                    status=data.status,
                )
                results["create"] += 1
            elif op.action == "update":
                if not op.data:
                    raise HTTPException(status_code=400, detail="update 작업에는 data가 필요합니다.")
                data = op.data
                listings_payload = {
                    language: listing.model_dump()
                    for language, listing in data.listings.items()
                }
                update_managed_inapp(
                    sku=data.sku,
                    default_language=data.default_language,
                    status=data.status,
                    default_price=data.default_price.model_dump(),
                    listings=listings_payload,
                    prices=data.prices,
                )
                results["update"] += 1
            elif op.action == "delete":
                delete_inapp_product(sku=op.sku)
                results["delete"] += 1
            else:  # pragma: no cover - defensive
                raise HTTPException(status_code=400, detail=f"지원하지 않는 작업 유형입니다: {op.action}")
    except HTTPException:
        raise
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc
    except Exception as exc:
        logger.exception("Failed to apply import operations")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"status": "ok", "summary": results}


@app.get("/health")
async def health_check():
    return {"status": "ok"}
