import logging
import os
from typing import Any, Dict, Optional

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


def create_managed_inapp(
    *,
    sku: str,
    default_language: str,
    default_title: str,
    default_description: str,
    price_won: Optional[int] = None,
    pricing_template_id: Optional[str] = None,
    translations: Optional[list[dict[str, str]]] = None,
) -> Dict[str, Any]:
    if not pricing_template_id:
        if price_won is None:
            raise ValueError("가격 또는 가격 템플릿 중 하나는 반드시 지정해야 합니다.")
        if price_won <= 0:
            raise ValueError("가격은 양수여야 합니다.")
        price_micros = price_won * 1_000_000
    service = _get_service()
    package_name = _get_package_name()
    body = {
        "packageName": package_name,
        "sku": sku,
        "status": "active",
        "purchaseType": "managedUser",
        "defaultLanguage": default_language,
        "defaultTitle": default_title,
        "defaultDescription": default_description,
    }
    if pricing_template_id:
        body["pricingTemplateId"] = pricing_template_id
    else:
        body["defaultPrice"] = {
            "priceMicros": str(price_micros),
            "currency": "KRW",
        }

    if translations:
        listings: Dict[str, Dict[str, str]] = {}
        for item in translations:
            language = item.get("language")
            title = item.get("title")
            description = item.get("description")
            if not language or not title or not description:
                raise ValueError("모든 번역 항목에는 언어, 이름, 설명이 필요합니다.")
            listings[language] = {"title": title, "description": description}
        if listings:
            body["listings"] = listings
    try:
        logger.info(
            "Creating managed in-app product",
            extra={
                "sku": sku,
                "default_language": default_language,
                "using_pricing_template": bool(pricing_template_id),
            },
        )
        response = service.inappproducts().insert(packageName=package_name, body=body).execute()
        logger.debug("Created in-app product response: %s", response)
        return response
    except HttpError as exc:
        logger.exception("Google API error while creating in-app product")
        raise RuntimeError(f"Google API 오류: {exc}") from exc
