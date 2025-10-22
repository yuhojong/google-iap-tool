import logging
import os
from typing import Any, Dict, Iterable, List, Optional

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

ANDROID_PUBLISHER_SCOPE = "https://www.googleapis.com/auth/androidpublisher"


def _get_package_name() -> str:
    package_name = os.getenv("PACKAGE_NAME")
    if not package_name:
        raise RuntimeError("PACKAGE_NAME 환경 변수가 설정되어 있지 않습니다.")
    logger.debug("Using package name: %s", package_name)
    return package_name


def _get_credentials() -> Credentials:
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 설정되어 있지 않습니다.")
    logger.debug("Loading credentials from: %s", credentials_path)
    return Credentials.from_service_account_file(credentials_path, scopes=[ANDROID_PUBLISHER_SCOPE])


def _get_service():
    creds = _get_credentials()
    logger.info("Initializing Google Play Android Publisher service client")
    return build("androidpublisher", "v3", credentials=creds, cache_discovery=False)


def list_inapp_products(page_token: Optional[str] = None) -> Dict[str, Any]:
    service = _get_service()
    package_name = _get_package_name()
    params: Dict[str, Any] = {"packageName": package_name}
    if page_token:
        params["token"] = page_token
    try:
        logger.info("Listing in-app products", extra={"page_token": page_token})
        response = service.inappproducts().list(**params).execute()
        logger.debug("Received list response: %s", response)
        return {
            "items": response.get("inappproduct", []),
            "nextPageToken": response.get("tokenPagination", {}).get("nextPageToken"),
        }
    except HttpError as exc:
        logger.exception("Google API error while listing in-app products")
        raise RuntimeError(f"Google API 오류: {exc}") from exc


def iterate_all_inapp_products() -> Iterable[Dict[str, Any]]:
    """Yield all in-app products by traversing every page."""

    next_token: Optional[str] = None
    while True:
        result = list_inapp_products(page_token=next_token)
        for item in result.get("items", []):
            yield item
        next_token = result.get("nextPageToken")
        if not next_token:
            break


def get_all_inapp_products() -> List[Dict[str, Any]]:
    return list(iterate_all_inapp_products())


def create_managed_inapp(
    *,
    sku: str,
    default_language: str,
    price_won: Optional[int] = None,
    regional_pricing: Optional[Dict[str, Any]] = None,
    translations: Optional[list[dict[str, str]]] = None,
    default_price: Optional[Dict[str, Any]] = None,
    prices: Optional[Dict[str, Any]] = None,
    status: str = "active",
) -> Dict[str, Any]:
    resolved_default_price: Optional[Dict[str, Any]] = None
    resolved_prices: Optional[Dict[str, Any]] = None

    if default_price is not None:
        if regional_pricing is not None or price_won is not None:
            raise ValueError("직접 지정한 가격과 다른 가격 옵션을 동시에 사용할 수 없습니다.")
        resolved_default_price = default_price
        resolved_prices = prices
    elif regional_pricing is not None:
        resolved_default_price = regional_pricing.get("defaultPrice")
        if not isinstance(resolved_default_price, dict):
            raise ValueError("가격 템플릿에 기본 가격 정보가 없습니다.")
        resolved_prices = regional_pricing.get("prices")
        if resolved_prices is not None and not isinstance(resolved_prices, dict):
            raise ValueError("가격 템플릿의 지역 가격 정보가 잘못되었습니다.")
    else:
        if price_won is None or price_won <= 0:
            raise ValueError("가격은 양수여야 합니다.")
        price_micros = price_won * 1_000_000
        resolved_default_price = {
            "priceMicros": str(price_micros),
            "currency": "KRW",
        }

    if not isinstance(resolved_default_price, dict):
        raise ValueError("기본 가격 정보가 필요합니다.")
    if "priceMicros" not in resolved_default_price or "currency" not in resolved_default_price:
        raise ValueError("기본 가격 정보에 priceMicros와 currency가 필요합니다.")

    service = _get_service()
    package_name = _get_package_name()
    listings: Dict[str, Dict[str, str]] = {}
    default_listing: Optional[Dict[str, str]] = None

    for item in translations or []:
        language = item.get("language")
        title = item.get("title")
        description = item.get("description")
        if not language or not title or not description:
            raise ValueError("모든 번역 항목에는 언어, 이름, 설명이 필요합니다.")
        listings[language] = {"title": title, "description": description}
        if language == default_language:
            default_listing = listings[language]

    if not default_listing:
        raise ValueError("기본 언어 번역 정보를 찾을 수 없습니다.")
    body = {
        "packageName": package_name,
        "sku": sku,
        "status": status,
        "purchaseType": "managedUser",
        "defaultLanguage": default_language,
        "defaultPrice": resolved_default_price,
        "listings": listings,
    }
    if resolved_prices:
        body["prices"] = resolved_prices
    try:
        logger.info(
            "Creating managed in-app product",
            extra={
                "sku": sku,
                "default_language": default_language,
            },
        )
        response = service.inappproducts().insert(packageName=package_name, body=body).execute()
        logger.debug("Created in-app product response: %s", response)
        return response
    except HttpError as exc:
        logger.exception("Google API error while creating in-app product")
        raise RuntimeError(f"Google API 오류: {exc}") from exc


def update_managed_inapp(
    *,
    sku: str,
    default_language: str,
    status: str,
    default_price: Dict[str, Any],
    listings: Dict[str, Dict[str, str]],
    prices: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not default_price or "priceMicros" not in default_price or "currency" not in default_price:
        raise ValueError("기본 가격 정보가 필요합니다.")

    service = _get_service()
    package_name = _get_package_name()

    body: Dict[str, Any] = {
        "packageName": package_name,
        "sku": sku,
        "status": status,
        "purchaseType": "managedUser",
        "defaultLanguage": default_language,
        "defaultPrice": default_price,
        "listings": listings,
    }
    if prices:
        body["prices"] = prices

    try:
        logger.info("Updating managed in-app product", extra={"sku": sku})
        response = (
            service.inappproducts()
            .update(packageName=package_name, sku=sku, body=body)
            .execute()
        )
        logger.debug("Updated in-app product response: %s", response)
        return response
    except HttpError as exc:
        logger.exception("Google API error while updating in-app product")
        raise RuntimeError(f"Google API 오류: {exc}") from exc


def delete_inapp_product(*, sku: str) -> None:
    service = _get_service()
    package_name = _get_package_name()
    try:
        logger.info("Deleting in-app product", extra={"sku": sku})
        service.inappproducts().delete(packageName=package_name, sku=sku).execute()
    except HttpError as exc:
        logger.exception("Google API error while deleting in-app product")
        raise RuntimeError(f"Google API 오류: {exc}") from exc
