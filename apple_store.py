"""Apple App Store Connect integration utilities."""

from __future__ import annotations

import base64
import binascii
import datetime as _dt
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

try:  # pragma: no cover - import-time environment check
    import jwt as _jwt_module
except ImportError as _jwt_import_error:  # pragma: no cover - handled lazily
    _jwt_module = None  # type: ignore[assignment]
else:  # pragma: no cover - exercised when dependency is available
    _jwt_import_error = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_APPLE_API_BASE = os.getenv(
    "APP_STORE_API_BASE_URL", "https://api.appstoreconnect.apple.com/v1"
)
_PRICE_POINTS_CACHE_TTL = int(os.getenv("APP_STORE_PRICE_POINT_CACHE_TTL", "1800"))


class AppleStoreConfigError(RuntimeError):
    """Raised when required Apple configuration is missing."""


def _encode_jwt(payload: Dict[str, Any], private_key: str, headers: Dict[str, str]) -> str:
    if _jwt_module is None:
        raise AppleStoreConfigError(
            "PyJWT 라이브러리가 설치되어 있지 않습니다. requirements.txt의 의존성을 설치해 주세요."
        ) from _jwt_import_error

    if hasattr(_jwt_module, "encode"):
        token = _jwt_module.encode(
            payload, private_key, algorithm="ES256", headers=headers
        )
    elif hasattr(_jwt_module, "JWT") and hasattr(_jwt_module, "jwk_from_pem"):
        jwt_instance = _jwt_module.JWT()
        jwk_key = _jwt_module.jwk_from_pem(private_key.encode("utf-8"))
        token = jwt_instance.encode(payload, jwk_key, alg="ES256", headers=headers)
    else:
        raise AppleStoreConfigError(
            "설치된 'jwt' 패키지가 PyJWT가 아니어서 토큰을 생성할 수 없습니다. "
            "'pip uninstall jwt' 후 'pip install PyJWT'를 실행해 주세요."
        )

    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise AppleStoreConfigError(
            f"환경 변수 '{name}'이(가) 설정되어 있지 않습니다."
        )
    return value


def _load_private_key() -> str:
    path = _require_env("APP_STORE_PRIVATE_KEY_PATH")
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return fp.read()
    except OSError as exc:  # pragma: no cover - depends on environment
        raise AppleStoreConfigError(
            f"APP_STORE_PRIVATE_KEY_PATH에서 키를 읽을 수 없습니다: {exc}"
        ) from exc


def _generate_token() -> str:
    issuer_id = _require_env("APP_STORE_ISSUER_ID")
    key_id = _require_env("APP_STORE_KEY_ID")
    private_key = _load_private_key()

    now = int(time.time())
    payload = {
        "iss": issuer_id,
        "iat": now,
        "exp": now + 20 * 60,
        "aud": "appstoreconnect-v1",
    }
    return _encode_jwt(payload, private_key, {"kid": key_id, "typ": "JWT"})


def _auth_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {_generate_token()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _request(
    method: str,
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not path.startswith("/"):
        path = "/" + path
    url = _APPLE_API_BASE.rstrip("/") + path
    logger.debug("Apple API Request %s %s", method, url)
    response = requests.request(
        method,
        url,
        headers=_auth_headers(),
        params=params,
        json=json,
        timeout=30,
    )
    if response.status_code >= 400:
        logger.error(
            "Apple API error", extra={"status": response.status_code, "body": response.text}
        )
        raise RuntimeError(
            f"Apple API 오류 {response.status_code}: {response.text.strip()}"
        )
    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def _extract_cursor(next_link: Optional[str]) -> Optional[str]:
    if not next_link:
        return None
    # The next link may include many query params; we only need the cursor value.
    if "page[cursor]=" not in next_link:
        return None
    return next_link.split("page[cursor]=", 1)[1].split("&", 1)[0]


def _get_app_id() -> str:
    return _require_env("APP_STORE_APP_ID")


def list_inapp_purchases(
    cursor: Optional[str] = None, limit: int = 200
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    params: Dict[str, Any] = {
        "include": "inAppPurchaseLocalizations,inAppPurchasePrices",
        "fields[inAppPurchases]": ",".join(
            [
                "productId",
                "referenceName",
                "inAppPurchaseType",
                "state",
                "clearedForSale",
                "familySharable",
            ]
        ),
        "page[limit]": limit,
    }
    if cursor:
        params["page[cursor]"] = cursor
    response = _request(
        "GET", f"/apps/{_get_app_id()}/inAppPurchases", params=params
    )
    included = response.get("included", [])
    localization_map = _index_included(included, "inAppPurchaseLocalizations")
    prices_map = _index_included(included, "inAppPurchasePrices")

    items: List[Dict[str, Any]] = []
    for record in response.get("data", []):
        items.append(
            _canonicalize_record(record, localization_map, prices_map)
        )

    next_cursor = _extract_cursor(response.get("links", {}).get("next"))
    return items, next_cursor


def iterate_all_inapp_purchases(limit: int = 200) -> Iterable[Dict[str, Any]]:
    cursor: Optional[str] = None
    while True:
        items, cursor = list_inapp_purchases(cursor=cursor, limit=limit)
        for item in items:
            yield item
        if not cursor:
            break


def get_all_inapp_purchases() -> List[Dict[str, Any]]:
    return list(iterate_all_inapp_purchases())


def _index_included(included: Iterable[Dict[str, Any]], resource_type: str) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for entry in included:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != resource_type:
            continue
        entry_id = entry.get("id")
        if not isinstance(entry_id, str):
            continue
        mapping[entry_id] = entry
    return mapping


def _canonicalize_record(
    record: Dict[str, Any],
    localization_map: Dict[str, Dict[str, Any]],
    prices_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    attributes = record.get("attributes") or {}
    relationships = record.get("relationships") or {}

    localizations: Dict[str, Dict[str, str]] = {}
    localization_relationship = relationships.get("inAppPurchaseLocalizations") or {}
    for loc in localization_relationship.get("data", []) or []:
        loc_id = loc.get("id")
        if not loc_id:
            continue
        payload = localization_map.get(loc_id, {}).get("attributes", {})
        locale = payload.get("locale")
        if not locale:
            continue
        localizations[locale] = {
            "name": payload.get("name", ""),
            "description": payload.get("description", ""),
        }

    prices: List[Dict[str, Any]] = []
    prices_relationship = relationships.get("inAppPurchasePrices") or {}
    for price_ref in prices_relationship.get("data", []) or []:
        price_id = price_ref.get("id")
        if not price_id:
            continue
        price_entry = prices_map.get(price_id) or {}
        price_attr = price_entry.get("attributes") or {}
        territory_attr = (
            (price_entry.get("relationships") or {})
            .get("territory", {})
            .get("data", {})
        )
        prices.append(
            {
                "id": price_id,
                "currency": price_attr.get("currency"),
                "price": price_attr.get("price"),
                "startDate": price_attr.get("startDate"),
                "territory": territory_attr.get("id"),
                "priceTier": price_attr.get("priceTier"),
            }
        )

    product_id = attributes.get("productId", "")
    return {
        "id": record.get("id", ""),
        "sku": product_id,
        "productId": product_id,
        "referenceName": attributes.get("referenceName", ""),
        "type": attributes.get("inAppPurchaseType", ""),
        "state": attributes.get("state", ""),
        "clearedForSale": attributes.get("clearedForSale", False),
        "familySharable": attributes.get("familySharable", False),
        "localizations": localizations,
        "prices": prices,
    }


def _format_iso8601(date: Optional[str]) -> Optional[str]:
    if not date:
        return None
    try:
        parsed = _dt.datetime.fromisoformat(date.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_localizations(
    localizations: Iterable[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for entry in localizations:
        if entry is None:
            continue
        if hasattr(entry, "model_dump"):
            data = entry.model_dump()  # type: ignore[attr-defined]
        elif isinstance(entry, dict):
            data = dict(entry)
        else:
            data = {
                key: getattr(entry, key)
                for key in ("locale", "name", "description", "review_screenshot")
                if hasattr(entry, key)
            }
        review = data.get("review_screenshot") or data.get("reviewScreenshot")
        review_payload = None
        if isinstance(review, dict):
            filename = review.get("filename") or review.get("fileName")
            content_type = review.get("content_type") or review.get("mimeType")
            encoded = review.get("data") or review.get("base64")
            if filename and encoded:
                review_payload = {
                    "filename": filename,
                    "content_type": content_type or "image/png",
                    "data": encoded,
                }
        normalized.append(
            {
                "locale": data.get("locale"),
                "name": data.get("name", ""),
                "description": data.get("description", ""),
                "review_screenshot": review_payload,
            }
        )
    return normalized


def _ensure_localizations(
    inapp_id: str,
    localizations: Iterable[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    normalized = _normalize_localizations(localizations)
    existing = _request(
        "GET", f"/inAppPurchases/{inapp_id}/inAppPurchaseLocalizations"
    ).get("data", [])
    existing_by_locale = {
        entry.get("attributes", {}).get("locale"): entry for entry in existing or []
    }

    desired_locales = {
        loc.get("locale") for loc in normalized if isinstance(loc.get("locale"), str)
    }

    # Remove localizations not present anymore
    for locale, entry in existing_by_locale.items():
        if locale not in desired_locales and entry.get("id"):
            _request("DELETE", f"/inAppPurchaseLocalizations/{entry['id']}")

    for loc in normalized:
        locale = loc.get("locale")
        if not locale:
            continue
        name = loc.get("name", "")
        description = loc.get("description", "")
        existing_entry = existing_by_locale.get(locale)
        payload = {
            "data": {
                "type": "inAppPurchaseLocalizations",
                "attributes": {
                    "name": name,
                    "description": description,
                    "locale": locale,
                },
                "relationships": {
                    "inAppPurchase": {"data": {"type": "inAppPurchases", "id": inapp_id}}
                },
            }
        }
        if existing_entry and existing_entry.get("id"):
            _request(
                "PATCH",
                f"/inAppPurchaseLocalizations/{existing_entry['id']}",
                json=payload,
            )
        else:
            _request("POST", "/inAppPurchaseLocalizations", json=payload)

    refreshed = _request(
        "GET", f"/inAppPurchases/{inapp_id}/inAppPurchaseLocalizations"
    ).get("data", [])
    locale_ids: Dict[str, str] = {}
    for entry in refreshed or []:
        entry_id = entry.get("id")
        attributes = entry.get("attributes") or {}
        locale = attributes.get("locale")
        if isinstance(locale, str) and entry_id:
            locale_ids[locale] = entry_id

    return normalized, locale_ids


def _build_price_point_cache_key(price_tier: str, territory: str) -> str:
    return f"{price_tier}:{territory.upper()}"


def _decode_screenshot_data(encoded: str) -> bytes:
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError("스크린샷 데이터를 디코딩할 수 없습니다.") from exc


def _get_existing_review_screenshot(localization_id: str) -> Optional[str]:
    response = _request(
        "GET",
        f"/inAppPurchaseLocalizations/{localization_id}/inAppPurchaseAppStoreReviewScreenshot",
    )
    data = response.get("data")
    if isinstance(data, dict):
        screenshot_id = data.get("id")
        if isinstance(screenshot_id, str):
            return screenshot_id
    return None


def _perform_upload_operations(operations: Iterable[Dict[str, Any]], data: bytes) -> None:
    for operation in operations or []:
        url = operation.get("url")
        method = operation.get("method") or "PUT"
        if not url:
            continue
        headers = {
            header.get("name"): header.get("value")
            for header in operation.get("requestHeaders") or []
            if header.get("name")
        }
        length = int(operation.get("length") or len(data))
        offset = int(operation.get("offset") or 0)
        chunk = data[offset : offset + length]
        response = requests.request(method, url, headers=headers, data=chunk, timeout=60)
        if response.status_code >= 400:
            raise RuntimeError(
                "스크린샷 업로드 중 오류가 발생했습니다: "
                f"{response.status_code} {response.text.strip()}"
            )


def _create_review_screenshot_record(
    localization_id: str, screenshot: Dict[str, Any]
) -> Dict[str, Any]:
    filename = screenshot.get("filename") or "review.png"
    content_type = screenshot.get("content_type") or "image/png"
    encoded = screenshot.get("data")
    if not isinstance(encoded, str):
        raise RuntimeError("스크린샷 데이터가 올바르지 않습니다.")
    binary = _decode_screenshot_data(encoded)
    checksum = hashlib.md5(binary).hexdigest()
    payload = {
        "data": {
            "type": "inAppPurchaseAppStoreReviewScreenshots",
            "attributes": {
                "fileName": filename,
                "fileSize": len(binary),
                "mimeType": content_type,
                "sourceFileChecksum": checksum,
                "uploaded": False,
            },
            "relationships": {
                "inAppPurchaseLocalization": {
                    "data": {
                        "type": "inAppPurchaseLocalizations",
                        "id": localization_id,
                    }
                }
            },
        }
    }
    response = _request("POST", "/inAppPurchaseAppStoreReviewScreenshots", json=payload)
    data = response.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("스크린샷 생성에 실패했습니다.")
    attributes = data.get("attributes") or {}
    upload_operations = attributes.get("uploadOperations") or []
    _perform_upload_operations(upload_operations, binary)
    screenshot_id = data.get("id")
    if screenshot_id:
        _request(
            "PATCH",
            f"/inAppPurchaseAppStoreReviewScreenshots/{screenshot_id}",
            json={
                "data": {
                    "type": "inAppPurchaseAppStoreReviewScreenshots",
                    "id": screenshot_id,
                    "attributes": {"uploaded": True},
                }
            },
        )
    return data


def _replace_review_screenshot(localization_id: str, screenshot: Dict[str, Any]) -> None:
    existing_id = _get_existing_review_screenshot(localization_id)
    if existing_id:
        _request("DELETE", f"/inAppPurchaseAppStoreReviewScreenshots/{existing_id}")
    _create_review_screenshot_record(localization_id, screenshot)


def _sync_review_screenshots(
    locale_ids: Dict[str, str], localizations: Iterable[Dict[str, Any]]
) -> None:
    for loc in localizations:
        locale = loc.get("locale")
        screenshot = loc.get("review_screenshot")
        if not locale or not screenshot:
            continue
        localization_id = locale_ids.get(locale)
        if not localization_id:
            continue
        try:
            _replace_review_screenshot(localization_id, screenshot)
        except Exception:
            logger.exception("Failed to upload review screenshot for locale %s", locale)
            raise


@dataclass
class _PricePointCacheEntry:
    expires_at: float
    result: Dict[str, Any]


_PRICE_POINT_CACHE: Dict[str, _PricePointCacheEntry] = {}


def _get_price_point(price_tier: str, territory: str) -> Dict[str, Any]:
    key = _build_price_point_cache_key(price_tier, territory)
    cached = _PRICE_POINT_CACHE.get(key)
    now = time.time()
    if cached and cached.expires_at > now:
        return cached.result

    params = {
        "filter[priceTier]": price_tier,
        "filter[territory]": territory,
        "page[limit]": 1,
    }
    response = _request("GET", "/inAppPurchasePricePoints", params=params)
    data = response.get("data", [])
    if not data:
        raise RuntimeError(
            f"가격 티어 '{price_tier}'에 대한 '{territory}' 가격 정보를 찾을 수 없습니다."
        )
    entry = data[0]
    _PRICE_POINT_CACHE[key] = _PricePointCacheEntry(
        expires_at=now + _PRICE_POINTS_CACHE_TTL,
        result=entry,
    )
    return entry


def _replace_price_schedule(
    inapp_id: str, price_tier: Optional[str], territory: str
) -> None:
    existing_prices = _request(
        "GET", f"/inAppPurchases/{inapp_id}/prices"
    ).get("data", [])
    for entry in existing_prices or []:
        entry_id = entry.get("id")
        if entry_id:
            _request("DELETE", f"/inAppPurchasePrices/{entry_id}")

    if not price_tier:
        return

    price_point = _get_price_point(price_tier, territory)
    attributes = price_point.get("attributes") or {}
    start_date = _format_iso8601(attributes.get("startDate"))
    payload = {
        "data": {
            "type": "inAppPurchasePrices",
            "attributes": {"startDate": start_date},
            "relationships": {
                "inAppPurchase": {
                    "data": {"type": "inAppPurchases", "id": inapp_id}
                },
                "inAppPurchasePricePoint": {
                    "data": {
                        "type": "inAppPurchasePricePoints",
                        "id": price_point.get("id"),
                    }
                },
            },
        }
    }
    _request("POST", "/inAppPurchasePrices", json=payload)


def create_inapp_purchase(
    *,
    product_id: str,
    reference_name: str,
    purchase_type: str,
    cleared_for_sale: bool,
    family_sharable: bool,
    review_note: Optional[str],
    price_tier: Optional[str],
    base_territory: str,
    localizations: Iterable[Dict[str, str]],
) -> Dict[str, Any]:
    payload = {
        "data": {
            "type": "inAppPurchases",
            "attributes": {
                "productId": product_id,
                "referenceName": reference_name,
                "clearedForSale": cleared_for_sale,
                "familySharable": family_sharable,
                "reviewNote": review_note,
                "inAppPurchaseType": purchase_type,
            },
            "relationships": {
                "apps": {
                    "data": [{"type": "apps", "id": _get_app_id()}]
                }
            },
        }
    }

    result = _request("POST", "/inAppPurchases", json=payload).get("data")
    if not result or not result.get("id"):
        raise RuntimeError("인앱 상품 생성에 실패했습니다.")

    inapp_id = result["id"]
    normalized_localizations, locale_ids = _ensure_localizations(
        inapp_id,
        localizations,
    )
    _sync_review_screenshots(locale_ids, normalized_localizations)
    _replace_price_schedule(inapp_id, price_tier, base_territory)

    refreshed = _request("GET", f"/inAppPurchases/{inapp_id}", params={"include": "inAppPurchaseLocalizations,inAppPurchasePrices"})
    localization_map = _index_included(
        refreshed.get("included", []), "inAppPurchaseLocalizations"
    )
    prices_map = _index_included(refreshed.get("included", []), "inAppPurchasePrices")
    return _canonicalize_record(refreshed.get("data", {}), localization_map, prices_map)


def update_inapp_purchase(
    *,
    inapp_id: str,
    reference_name: Optional[str],
    cleared_for_sale: Optional[bool],
    family_sharable: Optional[bool],
    review_note: Optional[str],
    price_tier: Optional[str],
    base_territory: str,
    localizations: Iterable[Dict[str, str]],
) -> Dict[str, Any]:
    attributes: Dict[str, Any] = {}
    if reference_name is not None:
        attributes["referenceName"] = reference_name
    if cleared_for_sale is not None:
        attributes["clearedForSale"] = cleared_for_sale
    if family_sharable is not None:
        attributes["familySharable"] = family_sharable
    if review_note is not None:
        attributes["reviewNote"] = review_note

    if attributes:
        payload = {
            "data": {
                "type": "inAppPurchases",
                "id": inapp_id,
                "attributes": attributes,
            }
        }
        _request("PATCH", f"/inAppPurchases/{inapp_id}", json=payload)

    normalized_localizations, locale_ids = _ensure_localizations(
        inapp_id,
        localizations,
    )
    _sync_review_screenshots(locale_ids, normalized_localizations)
    _replace_price_schedule(inapp_id, price_tier, base_territory)

    refreshed = _request("GET", f"/inAppPurchases/{inapp_id}", params={"include": "inAppPurchaseLocalizations,inAppPurchasePrices"})
    localization_map = _index_included(
        refreshed.get("included", []), "inAppPurchaseLocalizations"
    )
    prices_map = _index_included(refreshed.get("included", []), "inAppPurchasePrices")
    return _canonicalize_record(refreshed.get("data", {}), localization_map, prices_map)


def delete_inapp_purchase(inapp_id: str) -> None:
    _request("DELETE", f"/inAppPurchases/{inapp_id}")


def list_price_tiers(territory: str = "KOR") -> List[Dict[str, Any]]:
    tiers: Dict[str, Dict[str, Any]] = {}
    cursor: Optional[str] = None

    while True:
        params = {
            "filter[territory]": territory,
            "page[limit]": 200,
        }
        if cursor:
            params["page[cursor]"] = cursor
        response = _request("GET", "/inAppPurchasePricePoints", params=params)
        for entry in response.get("data", []):
            attributes = entry.get("attributes") or {}
            tier_id = attributes.get("priceTier")
            if not tier_id or tier_id in tiers:
                continue
            tiers[tier_id] = {
                "tier": tier_id,
                "currency": attributes.get("currency"),
                "customerPrice": attributes.get("customerPrice"),
                "proceeds": attributes.get("proceeds"),
            }
        cursor = _extract_cursor(response.get("links", {}).get("next"))
        if not cursor:
            break

    def _tier_sort_key(value: str) -> tuple[int, str]:
        digits = "".join(ch for ch in value if ch.isdigit())
        return (int(digits) if digits else 0, value)

    return [tiers[key] for key in sorted(tiers.keys(), key=_tier_sort_key)]
