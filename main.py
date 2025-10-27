import asyncio
import base64
import binascii
import csv
import hashlib
import hmac
import io
import json
import logging
import mimetypes
import os
import posixpath
import shlex
import subprocess
import zipfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from contextlib import asynccontextmanager
from functools import wraps

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from dotenv import load_dotenv

from apple_store import (
    AppleStoreConfigError,
    create_inapp_purchase as create_apple_inapp_purchase,
    delete_inapp_purchase as delete_apple_inapp_purchase,
    get_all_inapp_purchases,
    get_inapp_purchase_detail as get_apple_inapp_purchase_detail,
    get_fixed_price_territories,
    list_price_tiers as list_apple_price_tiers,
    update_inapp_purchase as update_apple_inapp_purchase,
)
from google_play import (
    create_managed_inapp,
    delete_inapp_product,
    get_all_inapp_products,
    update_managed_inapp,
)
from price_templates import (
    PriceTemplate,
    generate_price_templates_from_products,
    get_template_by_id,
    index_templates_by_price_micros,
)
from product_cache import (
    DEFAULT_PAGE_SIZE,
    delete_product as delete_cached_product,
    get_cached_products,
    get_paginated_products,
    refresh_products_from_remote,
    upsert_product as upsert_cached_product,
)

load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)

GOOGLE_STORE = "google"
APPLE_STORE = "apple"


def _parse_locale_list(value: str | None) -> List[str]:
    if not value:
        return []
    locales: List[str] = []
    for token in value.split(","):
        candidate = token.strip()
        if not candidate:
            continue
        if candidate not in locales:
            locales.append(candidate)
    return locales


APPLE_LOCALIZATION_LOCALES = _parse_locale_list(os.getenv("APPLE_LOCALIZATION_LOCALES"))
if not APPLE_LOCALIZATION_LOCALES:
    APPLE_LOCALIZATION_LOCALES = ["en-GB"]
elif "en-GB" not in APPLE_LOCALIZATION_LOCALES:
    APPLE_LOCALIZATION_LOCALES.insert(0, "en-GB")

APPLE_DEFAULT_LOCALIZATION_LOCALE = APPLE_LOCALIZATION_LOCALES[0]

app = FastAPI(title="iap-management-tool")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_files = StaticFiles(directory="static", html=True)

app.mount("/static", static_files, name="static")


CSV_PROCESSING_LOCK = asyncio.Lock()
CSV_IDLE_EVENT = asyncio.Event()
CSV_IDLE_EVENT.set()
_csv_processing_counter = 0

DEPLOYMENT_LOCK = asyncio.Lock()


async def _run_in_thread(func, /, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)

DEFAULT_REPO_PATH = Path(__file__).resolve().parent
TARGET_BRANCH = os.getenv("DEPLOY_TARGET_BRANCH", "main")
REPO_PATH = Path(os.getenv("DEPLOY_REPO_PATH", DEFAULT_REPO_PATH)).resolve()
RESTART_COMMAND = os.getenv("DEPLOY_RESTART_COMMAND")
POST_UPDATE_COMMANDS = os.getenv("DEPLOY_POST_UPDATE_COMMANDS")
GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET")


def _parse_additional_commands(value: str | None) -> List[List[str]]:
    commands: List[List[str]] = []
    if not value:
        return commands
    for line in value.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        commands.append(shlex.split(candidate))
    return commands


@asynccontextmanager
async def _csv_processing_guard():
    global _csv_processing_counter
    async with CSV_PROCESSING_LOCK:
        _csv_processing_counter += 1
        CSV_IDLE_EVENT.clear()
    try:
        yield
    finally:
        async with CSV_PROCESSING_LOCK:
            _csv_processing_counter = max(0, _csv_processing_counter - 1)
            if _csv_processing_counter == 0:
                CSV_IDLE_EVENT.set()


def csv_processing_endpoint(endpoint):
    @wraps(endpoint)
    async def wrapper(*args, **kwargs):
        async with _csv_processing_guard():
            return await endpoint(*args, **kwargs)

    return wrapper


async def _run_command(command: List[str], *, cwd: Path) -> None:
    command_display = " ".join(command)

    def _execute():
        result = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout:
            logger.info("%s stdout: %s", command_display, result.stdout.strip())
        if result.stderr:
            logger.warning("%s stderr: %s", command_display, result.stderr.strip())
        if result.returncode != 0:
            raise RuntimeError(
                f"Command '{command_display}' failed with exit code {result.returncode}"
            )

    logger.info("Running command: %s", command_display)
    await asyncio.to_thread(_execute)


async def _perform_deployment(trigger_id: str | None) -> None:
    async with DEPLOYMENT_LOCK:
        if trigger_id:
            logger.info("Deployment triggered by %s", trigger_id)
        else:
            logger.info("Deployment triggered")

        await CSV_IDLE_EVENT.wait()
        logger.info("CSV processing complete. Proceeding with deployment.")

        if not REPO_PATH.exists():
            raise RuntimeError(f"Repository path does not exist: {REPO_PATH}")

        commands: List[List[str]] = [
            ["git", "fetch", "origin", TARGET_BRANCH],
            ["git", "reset", "--hard", f"origin/{TARGET_BRANCH}"],
        ]
        commands.extend(_parse_additional_commands(POST_UPDATE_COMMANDS))

        if RESTART_COMMAND:
            commands.append(shlex.split(RESTART_COMMAND))
        else:
            logger.warning(
                "DEPLOY_RESTART_COMMAND is not set. Deployment will not restart the service."
            )

        for command in commands:
            await _run_command(command, cwd=REPO_PATH)

        logger.info("Deployment finished successfully.")


def _verify_github_signature(secret: str, payload: bytes, signature_header: str | None) -> bool:
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    signature = signature_header.split("=", 1)[1]
    digest = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(digest, signature)


def _schedule_deployment(trigger_id: str | None) -> None:
    asyncio.create_task(_perform_deployment(trigger_id))


@app.post("/webhook/github")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    if not GITHUB_WEBHOOK_SECRET:
        logger.error("GITHUB_WEBHOOK_SECRET is not configured.")
        raise HTTPException(status_code=500, detail="Webhook secret is not configured.")

    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")
    if not _verify_github_signature(GITHUB_WEBHOOK_SECRET, body, signature):
        logger.warning("Received GitHub webhook with invalid signature.")
        raise HTTPException(status_code=401, detail="Invalid webhook signature.")

    event = request.headers.get("X-GitHub-Event", "")
    if event != "push":
        logger.info("Ignoring GitHub event: %s", event)
        return {"status": "ignored", "reason": "unsupported_event"}

    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("Failed to decode webhook payload: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc

    expected_ref = f"refs/heads/{TARGET_BRANCH}"
    ref = payload.get("ref")
    if ref != expected_ref:
        logger.info("Ignoring push to %s. Watching %s.", ref, expected_ref)
        return {"status": "ignored", "reason": "branch_mismatch"}

    delivery = request.headers.get("X-GitHub-Delivery")
    head_commit = payload.get("after") or (payload.get("head_commit") or {}).get("id")
    trigger_id = head_commit or delivery

    background_tasks.add_task(_schedule_deployment, trigger_id)

    return {"status": "queued"}


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


class BulkCreateOperation(BaseModel):
    action: Literal["create"] = Field(default="create")
    sku: str
    data: ImportProductPayload


class BulkCreateApplyRequest(BaseModel):
    operations: List[BulkCreateOperation]


class AppleReviewScreenshot(BaseModel):
    filename: str = Field(..., min_length=1)
    content_type: str = Field(..., min_length=1)
    data: str = Field(..., min_length=1)

    @field_validator("data")
    @classmethod
    def validate_base64(cls, value: str) -> str:
        try:
            base64.b64decode(value, validate=True)
        except binascii.Error as exc:
            raise ValueError("스크린샷 데이터는 base64 문자열이어야 합니다.") from exc
        return value

    def decode_bytes(self) -> bytes:
        return base64.b64decode(self.data)


class AppleLocalization(BaseModel):
    locale: str = Field(..., min_length=2)
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    review_screenshot: Optional[AppleReviewScreenshot] = None


class AppleCreateInAppRequest(BaseModel):
    product_id: str = Field(..., min_length=1)
    reference_name: str = Field(..., min_length=1)
    purchase_type: Literal[
        "consumable",
        "nonConsumable",
        "nonRenewingSubscription",
        "autoRenewableSubscription",
    ]
    cleared_for_sale: bool = Field(default=True)
    family_sharable: bool = Field(default=False)
    review_note: Optional[str] = Field(default=None, max_length=4000)
    price_tier: Optional[str] = Field(default=None, min_length=1)
    base_territory: str = Field(default="KOR", min_length=2)
    localizations: List[AppleLocalization] = Field(default_factory=list)

    @model_validator(mode="after")
    def ensure_localizations(cls, model: "AppleCreateInAppRequest"):
        if not model.localizations:
            raise ValueError("최소 1개의 현지화 정보를 입력해야 합니다.")
        return model


class AppleUpdateInAppRequest(BaseModel):
    reference_name: Optional[str] = Field(default=None, min_length=1)
    cleared_for_sale: Optional[bool] = None
    family_sharable: Optional[bool] = None
    review_note: Optional[str] = Field(default=None, max_length=4000)
    price_tier: Optional[str] = Field(default=None, min_length=1)
    base_territory: str = Field(default="KOR", min_length=2)
    localizations: List[AppleLocalization] = Field(default_factory=list)


class AppleBulkDeletePreviewRequest(BaseModel):
    identifier_type: Literal["reference_name", "product_id", "sku", "iap_id"]
    values: List[str] = Field(..., min_length=1)

    @field_validator("values", mode="before")
    @classmethod
    def _coerce_values(cls, value):
        if isinstance(value, str):
            tokens = [token.strip() for token in value.replace(",", "\n").splitlines()]
            return [token for token in tokens if token]
        if isinstance(value, list):
            result = []
            for item in value:
                if isinstance(item, str):
                    token = item.strip()
                    if token:
                        result.append(token)
            return result
        raise TypeError("values must be a string or list of strings")

    @model_validator(mode="after")
    def _ensure_values(self):  # type: ignore[override]
        unique = list(dict.fromkeys(self.values))
        if not unique:
            raise ValueError("삭제할 항목을 하나 이상 입력해주세요.")
        object.__setattr__(self, "values", unique)
        return self


class AppleBulkDeleteItem(BaseModel):
    inapp_id: str = Field(..., min_length=1)
    product_id: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _normalize(self):  # type: ignore[override]
        object.__setattr__(self, "inapp_id", self.inapp_id.strip())
        object.__setattr__(self, "product_id", self.product_id.strip())
        if not self.inapp_id or not self.product_id:
            raise ValueError("유효한 상품 정보를 입력해주세요.")
        return self


class AppleBulkDeleteApplyRequest(BaseModel):
    items: List[AppleBulkDeleteItem] = Field(..., min_length=1)


class AppleImportPayload(BaseModel):
    product_id: str = Field(..., min_length=1)
    reference_name: str = Field(..., min_length=1)
    purchase_type: Optional[
        Literal[
            "consumable",
            "nonConsumable",
            "nonRenewingSubscription",
            "autoRenewableSubscription",
        ]
    ] = None
    cleared_for_sale: bool = Field(default=True)
    family_sharable: bool = Field(default=False)
    review_note: Optional[str] = Field(default=None, max_length=4000)
    price_tier: Optional[str] = Field(default=None, min_length=1)
    base_territory: str = Field(default="KOR", min_length=2)
    localizations: List[AppleLocalization] = Field(default_factory=list)


class AppleImportOperation(BaseModel):
    action: Literal["create", "update", "delete"]
    product_id: str = Field(..., min_length=1)
    data: Optional[AppleImportPayload] = None


class AppleImportApplyRequest(BaseModel):
    operations: List[AppleImportOperation]


def _collect_languages_from_products(products: Iterable[Dict[str, Any]]) -> List[str]:
    languages: set[str] = set()
    for item in products:
        listings = item.get("listings") or {}
        for language in listings.keys():
            if isinstance(language, str):
                languages.add(language)
    return sorted(languages)


def _canonicalize_google_product(item: Dict[str, Any]) -> Dict[str, Any]:
    listings = item.get("listings") or {}
    normalized_listings: Dict[str, Dict[str, str]] = {}
    for language, listing in listings.items():
        if not isinstance(language, str) or not isinstance(listing, dict):
            continue
        title = (listing.get("title") or "").strip()
        description = (listing.get("description") or "").strip()
        if not title and not description:
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


def _fetch_google_products() -> List[Dict[str, Any]]:
    return get_all_inapp_products()


def _fetch_apple_products() -> List[Dict[str, Any]]:
    products = get_all_inapp_purchases(include_relationships=False)
    for item in products:
        if isinstance(item, dict):
            item.pop("localizations", None)
    return products


def _collect_locales_from_apple_products(products: Iterable[Dict[str, Any]]) -> List[str]:
    locales: set[str] = set()
    for product in products:
        localization_map = product.get("localizations") or {}
        for locale in localization_map.keys():
            if isinstance(locale, str) and locale:
                locales.add(locale)
    return sorted(locales)


def _to_apple_summary(product: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(product, dict):
        return {}
    summary = {
        "id": product.get("id"),
        "resourceType": product.get("resourceType"),
        "sku": product.get("productId") or product.get("sku"),
        "productId": product.get("productId") or product.get("sku"),
        "referenceName": product.get("referenceName"),
        "type": product.get("type"),
        "state": product.get("state"),
        "clearedForSale": product.get("clearedForSale"),
        "familySharable": product.get("familySharable"),
    }
    prices = product.get("prices")
    if isinstance(prices, list):
        summary["prices"] = prices
    price_tier = product.get("priceTier")
    if price_tier and "prices" not in summary:
        summary["prices"] = [{"priceTier": price_tier}]
    return summary


def _normalize_zip_path(name: str) -> str:
    normalized = posixpath.normpath(name.replace("\\", "/")).strip("/")
    return normalized


def _detect_content_type(filename: str) -> str:
    guessed, _ = mimetypes.guess_type(filename)
    if guessed:
        return guessed
    return "application/octet-stream"


def _extract_csv_bundle(content: bytes, filename: str | None) -> Tuple[bytes, Dict[str, Dict[str, Any]]]:
    if not filename or not filename.lower().endswith(".zip"):
        return content, {}

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            csv_candidates: List[Tuple[str, bytes]] = []
            attachments: Dict[str, Dict[str, Any]] = {}
            for entry in archive.infolist():
                if entry.is_dir():
                    continue
                entry_name = _normalize_zip_path(entry.filename)
                try:
                    data = archive.read(entry)
                except KeyError as exc:
                    raise HTTPException(status_code=400, detail=f"ZIP 파일에서 {entry.filename} 을(를) 읽을 수 없습니다.") from exc
                if entry_name.lower().endswith(".csv"):
                    csv_candidates.append((entry_name, data))
                    continue
                attachment = {
                    "filename": os.path.basename(entry_name) or entry_name,
                    "content_type": _detect_content_type(entry_name),
                    "data": data,
                }
                keys = {
                    entry_name,
                    entry_name.lower(),
                    os.path.basename(entry_name),
                    os.path.basename(entry_name).lower(),
                }
                for key in keys:
                    if key:
                        attachments.setdefault(key, attachment)
            if not csv_candidates:
                raise HTTPException(status_code=400, detail="ZIP 파일에서 CSV를 찾을 수 없습니다.")
            csv_candidates.sort(key=lambda item: (item[0].count("/"), len(item[0])))
            return csv_candidates[0][1], attachments
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="유효한 ZIP 파일이 아닙니다.") from exc

    raise HTTPException(status_code=400, detail="ZIP 파일을 처리할 수 없습니다.")


def _resolve_attachment(
    value: str,
    attachments: Dict[str, Dict[str, Any]] | None,
) -> Dict[str, Any]:
    if not value:
        raise HTTPException(status_code=400, detail="스크린샷 파일명을 입력해주세요.")
    if not attachments:
        raise HTTPException(
            status_code=400,
            detail=f"스크린샷 파일 '{value}' 을(를) 찾을 수 없습니다. ZIP 파일에 이미지를 포함했는지 확인해주세요.",
        )
    normalized = _normalize_zip_path(value)
    candidates = [
        normalized,
        normalized.lower(),
        os.path.basename(normalized),
        os.path.basename(normalized).lower(),
    ]
    for key in candidates:
        if key in attachments:
            return attachments[key]
    raise HTTPException(
        status_code=400,
        detail=f"스크린샷 파일 '{value}' 을(를) ZIP 파일에서 찾을 수 없습니다.",
    )


def _parse_bool_cell(value: str, *, default: bool = False) -> bool:
    text = (value or "").strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f"}:
        return False
    raise ValueError("불리언 필드에는 true/false 값을 입력해주세요.")


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


def _parse_apple_import_csv(
    content: bytes, attachments: Dict[str, Dict[str, Any]] | None = None
) -> tuple[List[Dict[str, Any]], List[str]]:
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="CSV 파일은 UTF-8 인코딩이어야 합니다.") from exc

    reader = csv.DictReader(io.StringIO(decoded))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV 헤더를 찾을 수 없습니다.")

    locales: set[str] = set()
    for name in reader.fieldnames:
        if name.startswith("name_"):
            locales.add(name.split("_", 1)[1])
        if name.startswith("description_"):
            locales.add(name.split("_", 1)[1])
        if name.startswith("screenshot_"):
            locales.add(name.split("_", 1)[1])

    rows: List[Dict[str, Any]] = []
    for index, row in enumerate(reader, start=2):
        normalized = {key.strip(): (value or "").strip() for key, value in (row or {}).items()}
        normalized["__row"] = str(index)
        screenshots: Dict[str, Dict[str, Any]] = {}
        for locale in locales:
            key = f"screenshot_{locale}"
            screenshot_value = normalized.get(key, "")
            if not screenshot_value:
                continue
            attachment = _resolve_attachment(screenshot_value, attachments)
            data_bytes = attachment["data"]
            if len(data_bytes) > 10 * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail=f"행 {index}: 스크린샷 '{screenshot_value}' 용량이 10MB를 초과합니다.",
                )
            content_type = attachment.get("content_type") or "application/octet-stream"
            if content_type not in {"image/png", "image/jpeg"}:
                raise HTTPException(
                    status_code=400,
                    detail=f"행 {index}: 스크린샷 '{screenshot_value}' 은 PNG 또는 JPG 파일이어야 합니다.",
                )
            encoded = base64.b64encode(data_bytes).decode("ascii")
            screenshots[locale] = {
                "filename": attachment.get("filename") or screenshot_value,
                "content_type": content_type,
                "data": encoded,
            }
        if screenshots:
            normalized["__screenshots"] = screenshots
        rows.append(normalized)

    return rows, sorted(locale for locale in locales if locale)


def _parse_bulk_create_csv(content: bytes) -> tuple[List[Dict[str, Any]], List[str]]:
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
        default_language = (row.get("default_language") or "").strip()
        price_won = (row.get("price_won") or "").strip()
        status = (row.get("status") or "").strip() or None
        row_index = (row.get("index") or "").strip() or None

        if not sku and not default_language and not price_won:
            continue

        if not sku:
            raise HTTPException(status_code=400, detail=f"{index} 행: sku 값을 입력해주세요.")
        if sku in seen_skus:
            raise HTTPException(status_code=400, detail=f"CSV에 중복된 SKU가 있습니다: {sku}")
        seen_skus.add(sku)

        listings: Dict[str, Dict[str, str]] = {}
        for language in languages:
            title_key = f"title_{language}"
            description_key = f"description_{language}"
            title = (row.get(title_key) or "").strip()
            description = (row.get(description_key) or "").strip()
            if title or description:
                listings[language] = {"title": title, "description": description}

        rows.append(
            {
                "row": index,
                "index": row_index,
                "sku": sku,
                "status": status,
                "default_language": default_language,
                "price_won": price_won,
                "listings": listings,
            }
        )

    if not rows:
        raise HTTPException(status_code=400, detail="등록할 항목이 없습니다.")

    return rows, languages


def _build_bulk_create_operations(
    rows: List[Dict[str, Any]],
    existing_products: Dict[str, Dict[str, Any]],
    templates_by_price_micros: Dict[str, PriceTemplate] | None = None,
) -> Dict[str, Any]:
    templates_by_price_micros = templates_by_price_micros or {}
    operations: List[Dict[str, Any]] = []
    summary = {"create": 0}

    existing_skus = {sku for sku, data in existing_products.items() if sku}

    for entry in rows:
        sku = entry["sku"]
        if sku in existing_skus:
            raise HTTPException(status_code=400, detail=f"{sku} 는 이미 등록된 상품입니다.")

        default_language = entry["default_language"] or ""
        if not default_language:
            raise HTTPException(status_code=400, detail=f"{sku} 행: default_language 값을 입력해주세요.")

        price_won_value = entry["price_won"] or ""
        if not price_won_value:
            raise HTTPException(status_code=400, detail=f"{sku} 행: price_won 값을 입력해주세요.")

        try:
            price_micros, _ = _normalize_price_won(price_won_value)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"{sku} 행: {exc}") from exc

        listings = entry["listings"] or {}
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
            "status": entry.get("status") or "active",
            "default_language": default_language,
            "default_price": {
                "priceMicros": price_micros,
                "currency": "KRW",
            },
            "listings": listings,
            "prices": regional_prices,
        }

        operations.append({"action": "create", "sku": sku, "data": new_product, "index": entry.get("index")})
        summary["create"] += 1

    return {"operations": operations, "summary": summary}


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

        current_default_price = current.get("default_price") or {}
        current_price_micros = str(current_default_price.get("priceMicros") or "")
        current_currency = (current_default_price.get("currency") or "KRW").upper()

        applied_template = False
        template_prices: Optional[Dict[str, Any]] = None
        price_micros = current_price_micros
        currency = current_currency or "KRW"

        if entry["price_won"]:
            try:
                normalized_price_micros, _ = _normalize_price_won(entry["price_won"])
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"{sku} 행: {exc}") from exc
            price_micros = normalized_price_micros
            currency = "KRW"

            same_default_price = (
                current_price_micros
                and normalized_price_micros == current_price_micros
                and current_currency == "KRW"
            )

            if not same_default_price:
                template = templates_by_price_micros.get(normalized_price_micros)
            else:
                template = None

            if template:
                pricing_payload = template.to_pricing_payload()
                template_prices = pricing_payload.get("prices")
                applied_template = True
        else:
            price_micros = current_price_micros
            currency = current_currency or "KRW"

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


def _apple_row_to_payload(
    row: Dict[str, Any], locales: Iterable[str]
) -> AppleImportPayload:
    try:
        cleared_for_sale = _parse_bool_cell(row.get("cleared_for_sale", ""), default=True)
        family_sharable = _parse_bool_cell(row.get("family_sharable", ""), default=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"행 {row.get('__row')}: {exc}") from exc

    base_territory = row.get("base_territory") or "KOR"
    localizations: List[AppleLocalization] = []
    screenshots_map = row.get("__screenshots") or {}
    for locale in locales:
        name = row.get(f"name_{locale}") or ""
        description = row.get(f"description_{locale}") or ""
        if not name and not description:
            continue
        screenshot_payload = None
        raw_screenshot = screenshots_map.get(locale)
        if raw_screenshot:
            try:
                screenshot_payload = AppleReviewScreenshot(**raw_screenshot)
            except ValidationError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"행 {row.get('__row')}: {locale} 로케일의 스크린샷 정보를 확인할 수 없습니다.",
                ) from exc
        localizations.append(
            AppleLocalization(
                locale=locale,
                name=name or "",
                description=description or "",
                review_screenshot=screenshot_payload,
            )
        )

    payload = AppleImportPayload(
        product_id=(row.get("product_id") or "").strip(),
        reference_name=(row.get("reference_name") or "").strip(),
        purchase_type=(row.get("purchase_type") or "").strip() or None,
        cleared_for_sale=cleared_for_sale,
        family_sharable=family_sharable,
        review_note=(row.get("review_note") or "").strip() or None,
        price_tier=(row.get("price_tier") or "").strip() or None,
        base_territory=base_territory,
        localizations=localizations,
    )
    return payload


def _build_apple_import_operations(
    rows: List[Dict[str, str]], existing_products: Dict[str, Dict[str, Any]], locales: List[str]
) -> Dict[str, Any]:
    operations: List[Dict[str, Any]] = []
    summary = {"create": 0, "update": 0, "delete": 0}

    for row in rows:
        row_number = row.get("__row", "?")
        product_id = (row.get("product_id") or "").strip()
        if not product_id:
            raise HTTPException(status_code=400, detail=f"행 {row_number}: product_id가 필요합니다.")

        action_text = (row.get("action") or "").strip().lower()
        existing = existing_products.get(product_id)

        if action_text == "delete":
            operations.append(
                {
                    "action": "delete",
                    "product_id": product_id,
                    "current": existing,
                }
            )
            summary["delete"] += 1
            continue

        payload = _apple_row_to_payload(row, locales)
        if not payload.reference_name:
            raise HTTPException(
                status_code=400,
                detail=f"행 {row_number}: reference_name 값을 입력해주세요.",
            )
        if not payload.localizations:
            raise HTTPException(
                status_code=400,
                detail=f"행 {row_number}: 최소 1개의 로컬라이제이션이 필요합니다.",
            )

        if action_text == "create" or (action_text != "update" and product_id not in existing_products):
            if payload.purchase_type is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"행 {row_number}: create 작업에는 purchase_type이 필요합니다.",
                )
            operations.append(
                {
                    "action": "create",
                    "product_id": product_id,
                    "data": payload.model_dump(),
                }
            )
            summary["create"] += 1
        else:
            if existing is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"행 {row_number}: 업데이트 대상 상품을 찾을 수 없습니다.",
                )
            operations.append(
                {
                    "action": "update",
                    "product_id": product_id,
                    "data": payload.model_dump(),
                    "current": existing,
                }
            )
            summary["update"] += 1

    return {"operations": operations, "summary": summary}

@app.get("/api/google/inapp/list")
async def api_list_inapp(
    token: str | None = Query(default=None),
    refresh: bool = Query(default=False),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=200),
):
    try:
        if refresh:
            refresh_products_from_remote(GOOGLE_STORE, _fetch_google_products)
        items, next_token = get_paginated_products(
            GOOGLE_STORE,
            _fetch_google_products,
            token,
            page_size=page_size,
        )
        return {"items": items, "nextPageToken": next_token}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to list in-app products")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/google/inapp/export")
@csv_processing_endpoint
async def api_export_inapp() -> StreamingResponse:
    try:
        products = refresh_products_from_remote(GOOGLE_STORE, _fetch_google_products)
        languages = _collect_languages_from_products(products)
        output = io.StringIO()
        fieldnames = ["sku", "status", "default_language", "price_won"]
        for language in languages:
            fieldnames.append(f"title_{language}")
            fieldnames.append(f"description_{language}")
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for item in products:
            canonical = _canonicalize_google_product(item)
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


@app.get("/api/google/inapp/new/template")
@csv_processing_endpoint
async def api_bulk_create_template(
    row_count: int = Query(default=20, ge=1, le=500, description="템플릿에 포함할 행 수"),
) -> StreamingResponse:
    try:
        products = get_cached_products(GOOGLE_STORE, _fetch_google_products)
        languages = _collect_languages_from_products(products)
    except Exception as exc:
        logger.exception("Failed to prepare languages for bulk template", exc_info=exc)
        languages = []

    if not languages:
        languages = ["ko-KR"]

    fieldnames = ["index", "sku", "status", "default_language", "price_won"]
    for language in languages:
        fieldnames.append(f"title_{language}")
        fieldnames.append(f"description_{language}")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for idx in range(1, row_count + 1):
        writer.writerow({"index": idx})

    csv_bytes = output.getvalue().encode("utf-8-sig")
    headers = {"Content-Disposition": 'attachment; filename="iap-new-products-template.csv"'}
    return StreamingResponse(iter([csv_bytes]), media_type="text/csv", headers=headers)


@app.get("/api/google/pricing/templates")
async def api_list_price_templates():
    try:
        products = get_cached_products(GOOGLE_STORE, _fetch_google_products)
        templates = generate_price_templates_from_products(products)
        return {"templates": [template.to_response() for template in templates]}
    except Exception as exc:
        logger.exception("Failed to load price templates")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/google/inapp/new/import/preview")
@csv_processing_endpoint
async def api_bulk_create_preview(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="CSV 파일이 비어 있습니다.")

    rows, csv_languages = _parse_bulk_create_csv(content)

    try:
        existing_products_raw = refresh_products_from_remote(
            GOOGLE_STORE, _fetch_google_products
        )
        existing_products = {
            item.get("sku"): _canonicalize_google_product(item)
            for item in existing_products_raw
            if item.get("sku")
        }
        templates = generate_price_templates_from_products(existing_products_raw)
        templates_by_price = index_templates_by_price_micros(templates)
    except Exception as exc:
        logger.exception("Failed to prepare bulk create preview")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    operations_payload = _build_bulk_create_operations(rows, existing_products, templates_by_price)

    languages = sorted(set(csv_languages))

    return {
        "languages": languages,
        "operations": operations_payload["operations"],
        "summary": operations_payload["summary"],
    }


@app.post("/api/google/inapp/create")
async def api_create_inapp(payload: CreateInAppRequest):
    try:
        regional_pricing = None
        if payload.price_template_id:
            products = get_cached_products(GOOGLE_STORE, _fetch_google_products)
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
        upsert_cached_product(GOOGLE_STORE, created)
        return created
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to create managed in-app product")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/google/inapp/new/import/apply")
@csv_processing_endpoint
async def api_bulk_create_apply(request: BulkCreateApplyRequest):
    results = {"create": 0}

    try:
        existing_products = get_cached_products(GOOGLE_STORE, _fetch_google_products)
        existing_skus = {item.get("sku") for item in existing_products if item.get("sku")}

        for op in request.operations:
            data = op.data
            if data.sku in existing_skus:
                raise HTTPException(status_code=400, detail=f"{data.sku} 는 이미 등록된 상품입니다.")

            translations = [
                {"language": language, "title": listing.title, "description": listing.description}
                for language, listing in data.listings.items()
            ]
            created = create_managed_inapp(
                sku=data.sku,
                default_language=data.default_language,
                translations=translations,
                default_price=data.default_price.model_dump(),
                prices=data.prices,
                status=data.status,
            )
            upsert_cached_product(GOOGLE_STORE, created)
            existing_skus.add(data.sku)
            results["create"] += 1
    except HTTPException:
        raise
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc
    except Exception as exc:
        logger.exception("Failed to apply bulk create operations")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"status": "ok", "summary": results}


@app.put("/api/google/inapp/{sku}")
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
        updated = update_managed_inapp(
            sku=sku,
            default_language=payload.default_language,
            status=payload.status,
            default_price=payload.default_price.model_dump(),
            listings=listings_payload,
            prices=prices_payload,
        )
        upsert_cached_product(GOOGLE_STORE, updated)
        return {"status": "ok"}
    except HTTPException:
        raise
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc
    except Exception as exc:
        logger.exception("Failed to update managed in-app product")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/api/google/inapp/{sku}")
async def api_delete_inapp(sku: str):
    try:
        delete_inapp_product(sku=sku)
        delete_cached_product(GOOGLE_STORE, sku)
        return {"status": "ok"}
    except Exception as exc:
        logger.exception("Failed to delete in-app product")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/google/inapp/import/preview")
@csv_processing_endpoint
async def api_import_preview(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="CSV 파일이 비어 있습니다.")

    rows, csv_languages = _parse_import_csv(content)

    try:
        existing_products_raw = refresh_products_from_remote(
            GOOGLE_STORE, _fetch_google_products
        )
        existing_products = {
            item.get("sku"): _canonicalize_google_product(item)
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


@app.post("/api/google/inapp/import/apply")
@csv_processing_endpoint
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
                created = create_managed_inapp(
                    sku=data.sku,
                    default_language=data.default_language,
                    translations=translations,
                    default_price=data.default_price.model_dump(),
                    prices=data.prices,
                    status=data.status,
                )
                upsert_cached_product(GOOGLE_STORE, created)
                results["create"] += 1
            elif op.action == "update":
                if not op.data:
                    raise HTTPException(status_code=400, detail="update 작업에는 data가 필요합니다.")
                data = op.data
                listings_payload = {
                    language: listing.model_dump()
                    for language, listing in data.listings.items()
                }
                updated = update_managed_inapp(
                    sku=data.sku,
                    default_language=data.default_language,
                    status=data.status,
                    default_price=data.default_price.model_dump(),
                    listings=listings_payload,
                    prices=data.prices,
                )
                upsert_cached_product(GOOGLE_STORE, updated)
                results["update"] += 1
            elif op.action == "delete":
                delete_inapp_product(sku=op.sku)
                delete_cached_product(GOOGLE_STORE, op.sku)
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


@app.get("/api/apple/config")
async def api_apple_config():
    return {
        "localization": {
            "defaultLocale": APPLE_DEFAULT_LOCALIZATION_LOCALE,
            "locales": list(APPLE_LOCALIZATION_LOCALES),
        },
        "pricing": {
            "fixedPriceTerritories": list(get_fixed_price_territories()),
        },
    }


@app.get("/api/apple/inapp/list")
async def api_apple_list_inapp(
    token: str | None = Query(default=None),
    refresh: bool = Query(default=False),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=200),
):
    try:
        if refresh:
            await _run_in_thread(
                refresh_products_from_remote, APPLE_STORE, _fetch_apple_products
            )
        items, next_token = await _run_in_thread(
            get_paginated_products,
            APPLE_STORE,
            _fetch_apple_products,
            token,
            page_size=page_size,
        )
        return {"items": items, "nextPageToken": next_token}
    except HTTPException:
        raise
    except AppleStoreConfigError as exc:
        logger.error("Apple Store configuration error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to list Apple in-app purchases")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/apple/pricing/tiers")
async def api_apple_price_tiers(territory: str = Query(default="KOR", min_length=2)):
    try:
        tiers = await _run_in_thread(list_apple_price_tiers, territory)
        return {"tiers": tiers}
    except AppleStoreConfigError as exc:
        logger.error("Apple Store configuration error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to load Apple price tiers")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/apple/inapp/create")
async def api_apple_create_inapp(payload: AppleCreateInAppRequest):
    try:
        def _create_and_cache() -> Dict[str, Any]:
            result = create_apple_inapp_purchase(
                product_id=payload.product_id,
                reference_name=payload.reference_name,
                purchase_type=payload.purchase_type,
                cleared_for_sale=payload.cleared_for_sale,
                family_sharable=payload.family_sharable,
                review_note=payload.review_note,
                price_tier=payload.price_tier,
                base_territory=payload.base_territory,
                localizations=[loc.model_dump() for loc in payload.localizations],
            )
            upsert_cached_product(APPLE_STORE, _to_apple_summary(result))
            return result

        result = await _run_in_thread(_create_and_cache)
        return {"status": "ok", "item": result}
    except HTTPException:
        raise
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc
    except AppleStoreConfigError as exc:
        logger.error("Apple Store configuration error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to create Apple in-app purchase")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _find_apple_inapp(product_id: str) -> Dict[str, Any]:
    normalized = (product_id or "").strip()
    if not normalized:
        raise HTTPException(status_code=404, detail="Apple 인앱 상품을 찾을 수 없습니다.")

    def _lookup(products: Iterable[Dict[str, Any]]) -> Optional[str]:
        for item in products:
            if not isinstance(item, dict):
                continue
            candidate = item.get("productId") or item.get("sku")
            if candidate == normalized:
                inapp_id = item.get("id")
                if isinstance(inapp_id, str) and inapp_id:
                    return inapp_id
        return None

    products = get_cached_products(APPLE_STORE, _fetch_apple_products)
    inapp_id = _lookup(products)
    if not inapp_id:
        products = refresh_products_from_remote(APPLE_STORE, _fetch_apple_products)
        inapp_id = _lookup(products)

    if not inapp_id:
        raise HTTPException(status_code=404, detail="해당 Apple 인앱 상품을 찾을 수 없습니다.")

    try:
        return get_apple_inapp_purchase_detail(inapp_id)
    except AppleStoreConfigError as exc:
        logger.error("Apple Store configuration error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to fetch Apple in-app purchase detail")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/apple/inapp/{product_id}/detail")
async def api_apple_get_inapp_detail(product_id: str):
    item = await _run_in_thread(_find_apple_inapp, product_id)
    return {"item": item}


@app.put("/api/apple/inapp/{product_id}")
async def api_apple_update_inapp(product_id: str, payload: AppleUpdateInAppRequest):
    try:
        def _update_and_cache() -> Dict[str, Any]:
            target = _find_apple_inapp(product_id)
            inapp_id = target.get("id")
            if not inapp_id:
                raise HTTPException(status_code=404, detail="Apple 인앱 상품 ID를 확인할 수 없습니다.")

            localizations = payload.localizations or [
                AppleLocalization(locale=locale, name=data.get("name", ""), description=data.get("description", ""))
                for locale, data in (target.get("localizations") or {}).items()
            ]

            result = update_apple_inapp_purchase(
                inapp_id=inapp_id,
                reference_name=payload.reference_name,
                cleared_for_sale=payload.cleared_for_sale,
                family_sharable=payload.family_sharable,
                review_note=payload.review_note,
                price_tier=payload.price_tier,
                base_territory=payload.base_territory,
                localizations=[loc.model_dump() for loc in localizations],
            )
            upsert_cached_product(APPLE_STORE, _to_apple_summary(result))
            return result

        result = await _run_in_thread(_update_and_cache)
        return {"status": "ok", "item": result}
    except HTTPException:
        raise
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc
    except AppleStoreConfigError as exc:
        logger.error("Apple Store configuration error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to update Apple in-app purchase")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/api/apple/inapp/{product_id}")
async def api_apple_delete_inapp(product_id: str):
    try:
        def _delete_and_purge() -> None:
            target = _find_apple_inapp(product_id)
            inapp_id = target.get("id")
            if not inapp_id:
                raise HTTPException(status_code=404, detail="Apple 인앱 상품 ID를 확인할 수 없습니다.")
            delete_apple_inapp_purchase(inapp_id)
            delete_cached_product(APPLE_STORE, product_id)

        await _run_in_thread(_delete_and_purge)
        return {"status": "ok"}
    except HTTPException:
        raise
    except AppleStoreConfigError as exc:
        logger.error("Apple Store configuration error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to delete Apple in-app purchase")
        raise HTTPException(status_code=500, detail=str(exc)) from exc




@app.post("/api/apple/inapp/bulk-delete/preview")
async def api_apple_bulk_delete_preview(request: AppleBulkDeletePreviewRequest):
    try:
        products = await _run_in_thread(
            refresh_products_from_remote, APPLE_STORE, _fetch_apple_products
        )
    except AppleStoreConfigError as exc:
        logger.error("Apple Store configuration error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to load Apple in-app purchases for bulk delete preview")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    lookup_field = {
        "reference_name": "referenceName",
        "product_id": "productId",
        "sku": "productId",
        "iap_id": "id",
    }[request.identifier_type]

    index: Dict[str, List[Dict[str, Any]]] = {}
    for item in products:
        if not isinstance(item, dict):
            continue
        value = item.get(lookup_field)
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if not normalized:
            continue
        index.setdefault(normalized, []).append(item)

    matches: List[Dict[str, Any]] = []
    for raw in request.values:
        normalized = raw.strip()
        for item in index.get(normalized, []):
            summary = _to_apple_summary(item)
            if not summary:
                continue
            summary["matchedValue"] = normalized
            summary.setdefault("identifierType", request.identifier_type)
            matches.append(summary)

    return {"matches": matches}


@app.post("/api/apple/inapp/bulk-delete/apply")
async def api_apple_bulk_delete_apply(request: AppleBulkDeleteApplyRequest):
    try:
        def _apply_bulk_delete() -> Dict[str, Any]:
            summary = {"deleted": 0, "failed": []}

            for entry in request.items:
                try:
                    delete_apple_inapp_purchase(entry.inapp_id)
                except AppleStoreConfigError as exc:
                    logger.error("Apple Store configuration error: %s", exc)
                    raise HTTPException(status_code=503, detail=str(exc)) from exc
                except Exception as exc:
                    logger.exception(
                        "Failed to delete Apple in-app purchase %s", entry.inapp_id
                    )
                    summary["failed"].append(
                        {
                            "inapp_id": entry.inapp_id,
                            "product_id": entry.product_id,
                            "reason": str(exc),
                        }
                    )
                    continue

                delete_cached_product(APPLE_STORE, entry.product_id)
                summary["deleted"] += 1

            status = "ok" if not summary["failed"] else "partial"
            return {"status": status, "summary": summary}

        return await _run_in_thread(_apply_bulk_delete)
    except HTTPException:
        raise


@app.get("/health")
async def health_check():
    return {"status": "ok"}
