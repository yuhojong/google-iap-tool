"""Apple App Store Connect integration utilities."""

from __future__ import annotations

import datetime as _dt
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import threading

import jwt
import requests

logger = logging.getLogger(__name__)

_APPLE_API_BASE = os.getenv(
    "APP_STORE_API_BASE_URL", "https://api.appstoreconnect.apple.com/v1"
)
_PRICE_POINTS_CACHE_TTL = int(os.getenv("APP_STORE_PRICE_POINT_CACHE_TTL", "1800"))


@dataclass
class _CachedPrivateKey:
    path: str
    value: str
    mtime: float


_private_key_cache: Optional[_CachedPrivateKey] = None
_private_key_lock = threading.Lock()
_token_cache: Optional[Tuple[str, int]] = None
_token_lock = threading.Lock()
_session = requests.Session()
_session_lock = threading.Lock()
_cached_app_id: Optional[str] = os.getenv("APP_STORE_APP_ID")


class AppleStoreConfigError(RuntimeError):
    """Raised when required Apple configuration is missing."""


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
        stat_result = os.stat(path)
    except OSError as exc:  # pragma: no cover - depends on environment
        raise AppleStoreConfigError(
            f"APP_STORE_PRIVATE_KEY_PATH에서 키를 읽을 수 없습니다: {exc}"
        ) from exc

    with _private_key_lock:
        global _private_key_cache
        if (
            _private_key_cache
            and _private_key_cache.path == path
            and _private_key_cache.mtime == stat_result.st_mtime
        ):
            return _private_key_cache.value

        try:
            with open(path, "r", encoding="utf-8") as fp:
                key_value = fp.read()
        except OSError as exc:  # pragma: no cover - depends on environment
            raise AppleStoreConfigError(
                f"APP_STORE_PRIVATE_KEY_PATH에서 키를 읽을 수 없습니다: {exc}"
            ) from exc

        _private_key_cache = _CachedPrivateKey(
            path=path, value=key_value, mtime=stat_result.st_mtime
        )
        with _token_lock:
            global _token_cache
            _token_cache = None
        return key_value


def _generate_token() -> str:
    issuer_id = _require_env("APP_STORE_ISSUER_ID")
    key_id = _require_env("APP_STORE_KEY_ID")

    now = int(time.time())

    with _token_lock:
        global _token_cache
        if _token_cache and now < _token_cache[1] - 30:
            return _token_cache[0]

        private_key = _load_private_key()
        payload = {
            "iss": issuer_id,
            "iat": now,
            "exp": now + 20 * 60,
            "aud": "appstoreconnect-v1",
        }
        token = jwt.encode(
            payload,
            private_key,
            algorithm="ES256",
            headers={"kid": key_id, "typ": "JWT"},
        )
        if isinstance(token, bytes):
            token = token.decode("utf-8")
        _token_cache = (token, payload["exp"])
        return token


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
    try:
        with _session_lock:
            response = _session.request(
                method,
                url,
                headers=_auth_headers(),
                params=params,
                json=json,
                timeout=30,
            )
    except requests.RequestException as exc:
        logger.error("Apple API request failed", exc_info=exc)
        raise RuntimeError("Apple API 요청 중 네트워크 오류가 발생했습니다.") from exc
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
    global _cached_app_id
    if not _cached_app_id:
        _cached_app_id = _require_env("APP_STORE_APP_ID")
    return _cached_app_id


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


def _ensure_localizations(
    inapp_id: str,
    localizations: Iterable[Dict[str, str]],
) -> None:
    existing = _request(
        "GET", f"/inAppPurchases/{inapp_id}/inAppPurchaseLocalizations"
    ).get("data", [])
    existing_by_locale = {
        entry.get("attributes", {}).get("locale"): entry for entry in existing or []
    }

    desired_locales = {loc.get("locale") for loc in localizations if loc.get("locale")}

    # Remove localizations not present anymore
    for locale, entry in existing_by_locale.items():
        if locale not in desired_locales and entry.get("id"):
            _request("DELETE", f"/inAppPurchaseLocalizations/{entry['id']}")

    for loc in localizations:
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


def _build_price_point_cache_key(price_tier: str, territory: str) -> str:
    return f"{price_tier}:{territory.upper()}"


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
    _ensure_localizations(
        inapp_id,
        [
            {"locale": loc.get("locale"), "name": loc.get("name"), "description": loc.get("description")}
            for loc in localizations
        ],
    )
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

    _ensure_localizations(
        inapp_id,
        [
            {"locale": loc.get("locale"), "name": loc.get("name"), "description": loc.get("description")}
            for loc in localizations
        ],
    )
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
