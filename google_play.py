import os
from typing import Any, Dict, Optional

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

ANDROID_PUBLISHER_SCOPE = "https://www.googleapis.com/auth/androidpublisher"


def _get_package_name() -> str:
    package_name = os.getenv("PACKAGE_NAME")
    if not package_name:
        raise RuntimeError("PACKAGE_NAME 환경 변수가 설정되어 있지 않습니다.")
    return package_name


def _get_credentials() -> Credentials:
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 설정되어 있지 않습니다.")
    return Credentials.from_service_account_file(credentials_path, scopes=[ANDROID_PUBLISHER_SCOPE])


def _get_service():
    creds = _get_credentials()
    return build("androidpublisher", "v3", credentials=creds, cache_discovery=False)


def list_inapp_products(page_token: Optional[str] = None) -> Dict[str, Any]:
    service = _get_service()
    package_name = _get_package_name()
    params: Dict[str, Any] = {"packageName": package_name}
    if page_token:
        params["token"] = page_token
    try:
        response = service.inappproducts().list(**params).execute()
        return {
            "items": response.get("inappproduct", []),
            "nextPageToken": response.get("tokenPagination", {}).get("nextPageToken"),
        }
    except HttpError as exc:
        raise RuntimeError(f"Google API 오류: {exc}") from exc


def create_managed_inapp(sku: str, title: str, description: str, price_won: int) -> Dict[str, Any]:
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
        "defaultLanguage": "ko-KR",
        "defaultTitle": title,
        "defaultDescription": description,
        "defaultPrice": {
            "priceMicros": str(price_micros),
            "currency": "KRW",
        },
    }
    try:
        return service.inappproducts().insert(packageName=package_name, body=body).execute()
    except HttpError as exc:
        raise RuntimeError(f"Google API 오류: {exc}") from exc
