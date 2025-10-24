"""Apple App Store Connect integration utilities."""

from __future__ import annotations

import base64
import binascii
import datetime as _dt
import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

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

_TOKEN_LOCK = threading.Lock()
_TOKEN_CACHE: Optional[Tuple[str, int]] = None

#
# Some older versions of the App Store Connect API supported JSON:API style
# query parameters such as ``include`` and ``page[limit]`` when listing in-app
# purchases. The latest version rejects these parameters with ``PARAMETER_ERROR``
# responses. We optimistically attempt to use the richer query first and fall
# back to a compatibility mode when the server indicates that the parameters
# are no longer accepted.
#
_INAPP_LIST_SUPPORTS_EXTENDED_PARAMS = True
_INAPP_LIST_SUPPORTS_LIMIT_PARAM = True
_INAPP_PRICE_RELATIONSHIP_AVAILABLE = True


_PRICE_TIER_GUESS_RANGE = tuple(str(value) for value in range(0, 201))


class AppleStoreConfigError(RuntimeError):
    """Raised when required Apple configuration is missing."""


class AppleStorePermissionError(AppleStoreConfigError):
    """Raised when the App Store Connect API denies an operation."""


def _encode_jwt(payload: Dict[str, Any], private_key: str, headers: Dict[str, str]) -> str:
    if _jwt_module is None:
        raise AppleStoreConfigError(
            "PyJWT 라이브러리가 설치되어 있지 않습니다. requirements.txt의 의존성을 설치해 주세요."
        ) from _jwt_import_error

    encode_func = getattr(_jwt_module, "encode", None)
    if callable(encode_func):
        try:
            token = encode_func(payload, private_key, algorithm="ES256", headers=headers)
        except Exception as exc:  # pragma: no cover - delegates to dependency
            raise AppleStoreConfigError(
                "PyJWT로 JWT를 생성하는 데 실패했습니다. 비공개 키가 올바른지 확인해 주세요."
            ) from exc

        if isinstance(token, bytes):
            token = token.decode("utf-8")
        return token

    logger.warning(
        "PyJWT encode API를 사용할 수 없어 cryptography 기반 JWT 생성을 시도합니다."
    )
    return _encode_jwt_with_cryptography(payload, private_key, headers)


def _encode_jwt_with_cryptography(
    payload: Dict[str, Any], private_key: str, headers: Dict[str, str]
) -> str:
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise AppleStoreConfigError(
            "PyJWT 대신 다른 'jwt' 패키지를 사용하려면 'cryptography' 패키지가 필요합니다. "
            "'pip install PyJWT' 또는 'pip install cryptography'를 실행해 주세요."
        ) from exc

    try:
        private_key_obj = serialization.load_pem_private_key(
            private_key.encode("utf-8"), password=None
        )
    except (TypeError, ValueError) as exc:
        raise AppleStoreConfigError(
            "Apple API 비공개 키를 읽을 수 없습니다. 키 파일이 손상되지 않았는지 확인해 주세요."
        ) from exc

    if not isinstance(private_key_obj, ec.EllipticCurvePrivateKey):
        raise AppleStoreConfigError(
            "Apple API 비공개 키는 ES256(ECDSA, P-256) 키여야 합니다."
        )

    curve_name = getattr(private_key_obj.curve, "name", "")
    if curve_name not in {"secp256r1", "prime256v1"}:
        raise AppleStoreConfigError(
            "Apple API 비공개 키는 P-256 곡선을 사용해야 합니다."
        )

    header: Dict[str, Any] = {"alg": "ES256", **headers}
    header.setdefault("typ", "JWT")

    def _b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    signing_segments = []
    for segment in (header, payload):
        json_segment = json.dumps(segment, separators=(",", ":"), ensure_ascii=False)
        signing_segments.append(_b64url(json_segment.encode("utf-8")))

    signing_input = ".".join(signing_segments)

    signature_der = private_key_obj.sign(
        signing_input.encode("utf-8"), ec.ECDSA(hashes.SHA256())
    )
    r, s = decode_dss_signature(signature_der)
    size = (private_key_obj.key_size + 7) // 8
    signature = r.to_bytes(size, "big") + s.to_bytes(size, "big")
    signing_segments.append(_b64url(signature))
    return ".".join(signing_segments)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise AppleStoreConfigError(
            f"환경 변수 '{name}'이(가) 설정되어 있지 않습니다."
        )
    value = value.strip()
    if not value:
        raise AppleStoreConfigError(
            f"환경 변수 '{name}'이(가) 비어 있습니다."
        )
    return value


_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _require_uuid_env(name: str) -> str:
    value = _require_env(name)
    if not _UUID_RE.match(value):
        raise AppleStoreConfigError(
            f"환경 변수 '{name}' 값이 올바른 Issuer ID 형식(UUID)인지 확인해 주세요."
        )
    return value


_KEY_ID_RE = re.compile(r"^[A-Z0-9]{10}$")


def _require_key_id_env(name: str) -> str:
    value = _require_env(name)
    if not _KEY_ID_RE.match(value.upper()):
        raise AppleStoreConfigError(
            f"환경 변수 '{name}' 값이 올바른 Key ID 형식(대문자 영숫자 10자)인지 확인해 주세요."
        )
    return value.upper()


def _load_private_key() -> str:
    path = _require_env("APP_STORE_PRIVATE_KEY_PATH")
    try:
        with open(path, "r", encoding="utf-8") as fp:
            contents = fp.read()
    except OSError as exc:  # pragma: no cover - depends on environment
        raise AppleStoreConfigError(
            f"APP_STORE_PRIVATE_KEY_PATH에서 키를 읽을 수 없습니다: {exc}"
        ) from exc

    contents = contents.lstrip("\ufeff").strip()
    if "-----BEGIN" not in contents or "PRIVATE KEY-----" not in contents:
        raise AppleStoreConfigError(
            "Apple API 비공개 키 파일 형식이 올바르지 않습니다. App Store Connect에서 내려받은 .p8 파일인지 확인해 주세요."
        )
    return contents + ("\n" if not contents.endswith("\n") else "")


def _generate_token() -> str:
    global _TOKEN_CACHE
    issuer_id = _require_uuid_env("APP_STORE_ISSUER_ID")
    key_id = _require_key_id_env("APP_STORE_KEY_ID")
    private_key = _load_private_key()

    now = int(time.time())
    with _TOKEN_LOCK:
        cached = _TOKEN_CACHE
        if cached and now < cached[1] - 30:
            return cached[0]

        issued_at = now - 10  # Allow small clock skew for local environments
        expires_at = issued_at + 19 * 60  # Keep lifetime under Apple's 20 minute limit
        payload = {
            "iss": issuer_id,
            "iat": issued_at,
            "exp": expires_at,
            "aud": "appstoreconnect-v1",
        }
        token = _encode_jwt(payload, private_key, {"kid": key_id, "typ": "JWT"})
        _TOKEN_CACHE = (token, expires_at)
        return token


def _auth_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {_generate_token()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _invalidate_token_cache() -> None:
    global _TOKEN_CACHE
    with _TOKEN_LOCK:
        _TOKEN_CACHE = None


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
        body_text = response.text.strip()
        logger.error(
            "Apple API error", extra={"status": response.status_code, "body": body_text}
        )
        if response.status_code == 401:
            _invalidate_token_cache()
            raise AppleStoreConfigError(
                _format_authorization_error(body_text)
            )
        raise RuntimeError(
            f"Apple API 오류 {response.status_code}: {body_text}"
        )
    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def _format_authorization_error(body_text: str) -> str:
    guidance = (
        "Apple API 인증에 실패했습니다. Issuer ID, Key ID, 비공개 키 파일을 다시 확인하고 "
        "서버의 시스템 시간이 정확한지 검증해 주세요."
    )
    if not body_text:
        return guidance
    try:
        payload = json.loads(body_text)
    except json.JSONDecodeError:
        return f"{guidance} 원본 오류: {body_text}"

    errors = payload.get("errors")
    if isinstance(errors, list) and errors:
        entry = errors[0] or {}
        code = entry.get("code") or entry.get("status")
        detail = entry.get("detail") or entry.get("title")
        if code or detail:
            suffix = " ".join(filter(None, [f"[{code}]" if code else "", detail]))
            return f"{guidance} {suffix}".strip()
    return f"{guidance} 원본 오류: {body_text}"


def _extract_cursor(next_link: Optional[str]) -> Optional[str]:
    if not next_link:
        return None

    parsed = urlparse(next_link)
    query = parse_qs(parsed.query or "")
    for key in ("page[cursor]", "cursor"):
        values = query.get(key)
        if values:
            return values[0]
    # Some responses may include an already URL-encoded ``page[cursor]`` value
    # in the path portion (e.g. ``...page%5Bcursor%5D=...``). Fallback to the
    # original string search so we do not miss such cases.
    if "page[cursor]=" in next_link:
        return next_link.split("page[cursor]=", 1)[1].split("&", 1)[0]
    if "page%5Bcursor%5D=" in next_link:
        return next_link.split("page%5Bcursor%5D=", 1)[1].split("&", 1)[0]
    return None


def _get_app_id() -> str:
    return _require_env("APP_STORE_APP_ID")


def _is_parameter_error(exc: Exception) -> bool:
    message = str(exc)
    if not message:
        return False
    markers = (
        "PARAMETER_ERROR",
        "not a valid relationship name",
        "not a valid field name",
        "not be used with this request",
    )
    return any(marker in message for marker in markers)


def _is_path_error(exc: Exception) -> bool:
    message = str(exc)
    if not message:
        return False
    return "PATH_ERROR" in message or "The URL path is not valid" in message


def _is_forbidden_noop_error(exc: Exception) -> bool:
    message = str(exc)
    if not message:
        return False
    if "FORBIDDEN_ERROR" not in message:
        return False
    return "no allowed operations" in message.lower()


def _is_forbidden_error(exc: Exception) -> bool:
    message = str(exc)
    if not message:
        return False
    return "FORBIDDEN_ERROR" in message


def list_inapp_purchases(
    cursor: Optional[str] = None, limit: int = 200
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    global _INAPP_LIST_SUPPORTS_EXTENDED_PARAMS, _INAPP_LIST_SUPPORTS_LIMIT_PARAM

    endpoint = f"/apps/{_get_app_id()}/inAppPurchases"

    if _INAPP_LIST_SUPPORTS_EXTENDED_PARAMS:
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
        }
        if limit:
            params["page[limit]"] = limit
        if cursor:
            params["page[cursor]"] = cursor
        try:
            response = _request("GET", endpoint, params=params)
        except RuntimeError as exc:
            if _is_parameter_error(exc):
                logger.warning(
                    "Apple API rejected extended in-app purchase parameters; "
                    "retrying in compatibility mode."
                )
                _INAPP_LIST_SUPPORTS_EXTENDED_PARAMS = False
            else:
                raise
        else:
            included = response.get("included", [])
            localization_map = _index_included(
                included, "inAppPurchaseLocalizations"
            )
            prices_map = _index_included(included, "inAppPurchasePrices")

            items: List[Dict[str, Any]] = []
            for record in response.get("data", []):
                items.append(
                    _canonicalize_record(record, localization_map, prices_map)
                )

            next_cursor = _extract_cursor(response.get("links", {}).get("next"))
            return items, next_cursor

    params = {}
    if cursor:
        params["cursor"] = cursor
    if limit and _INAPP_LIST_SUPPORTS_LIMIT_PARAM:
        params["limit"] = limit

    try:
        response = _request("GET", endpoint, params=params)
    except RuntimeError as exc:
        if "limit" in params and _is_parameter_error(exc):
            logger.info(
                "Apple API rejected limit parameter when listing in-app purchases; "
                "retrying without limit."
            )
            _INAPP_LIST_SUPPORTS_LIMIT_PARAM = False
            params.pop("limit", None)
            response = _request("GET", endpoint, params=params)
        else:
            raise

    items = []
    for record in response.get("data", []):
        item = _canonicalize_record(record, {}, {})
        inapp_id = item.get("id")
        item["localizations"] = _fetch_inapp_localizations(inapp_id)
        item["prices"] = _fetch_inapp_prices(inapp_id)
        items.append(item)

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


def _parse_price_entry(entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict):
        return None

    attributes = entry.get("attributes") or {}
    relationships = entry.get("relationships") or {}
    territory = (
        (relationships.get("territory") or {})
        .get("data", {})
        .get("id")
    )

    return {
        "id": entry.get("id"),
        "currency": attributes.get("currency"),
        "price": attributes.get("price"),
        "startDate": attributes.get("startDate"),
        "territory": territory,
        "priceTier": attributes.get("priceTier"),
    }


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
        price_entry = prices_map.get(price_id) or price_ref
        parsed_price = _parse_price_entry(price_entry)
        if parsed_price:
            prices.append(parsed_price)

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


def _normalize_localization_entries(entries: object) -> List[Dict[str, Any]]:
    if isinstance(entries, list):
        return [entry for entry in entries if isinstance(entry, dict)]
    if isinstance(entries, dict):
        return [entries]
    return []


def _load_localization_entries_via_filter(inapp_id: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    while True:
        params: Dict[str, Any] = {
            "filter[inAppPurchase]": inapp_id,
            "limit": "200",
        }
        if cursor:
            params["cursor"] = cursor

        try:
            response = _request(
                "GET", "/inAppPurchaseLocalizations", params=params
            )
        except RuntimeError as exc:
            if (
                _is_parameter_error(exc)
                or _is_forbidden_noop_error(exc)
                or _is_forbidden_error(exc)
            ):
                logger.warning(
                    "Apple API denied localization filter lookup for %s", inapp_id
                )
                return entries
            raise

        entries.extend(_normalize_localization_entries(response.get("data")))

        cursor = _extract_cursor(response.get("links", {}).get("next"))
        if not cursor:
            break

    return entries


def _list_localization_entries(inapp_id: str) -> List[Dict[str, Any]]:
    try:
        response = _request(
            "GET", f"/inAppPurchases/{inapp_id}/inAppPurchaseLocalizations"
        )
    except RuntimeError as exc:
        if not (
            _is_path_error(exc)
            or _is_forbidden_noop_error(exc)
            or _is_forbidden_error(exc)
        ):
            raise
        logger.info(
            "Apple API reported missing in-app localization relationship; "
            "retrying with filtered localization lookup.",
        )
        return _load_localization_entries_via_filter(inapp_id)

    return _normalize_localization_entries(response.get("data"))


def _fetch_inapp_localizations(inapp_id: Optional[str]) -> Dict[str, Dict[str, str]]:
    if not inapp_id:
        return {}

    result: Dict[str, Dict[str, str]] = {}
    for entry in _list_localization_entries(inapp_id):
        attributes = entry.get("attributes") or {}
        locale = attributes.get("locale")
        if not locale:
            continue
        result[locale] = {
            "name": attributes.get("name", ""),
            "description": attributes.get("description", ""),
        }
    return result


def _fetch_inapp_prices(inapp_id: Optional[str]) -> List[Dict[str, Any]]:
    global _INAPP_PRICE_RELATIONSHIP_AVAILABLE

    if not inapp_id:
        return []

    if not _INAPP_PRICE_RELATIONSHIP_AVAILABLE:
        return []

    try:
        response = _request("GET", f"/inAppPurchases/{inapp_id}/prices")
    except RuntimeError as exc:
        if (
            _is_path_error(exc)
            or _is_forbidden_noop_error(exc)
            or _is_forbidden_error(exc)
        ):
            if _INAPP_PRICE_RELATIONSHIP_AVAILABLE:
                logger.warning(
                    "Apple API rejected in-app purchase price relationship lookup; "
                    "price data will not be included in responses."
                )
            _INAPP_PRICE_RELATIONSHIP_AVAILABLE = False
            return []
        raise

    prices: List[Dict[str, Any]] = []
    for entry in response.get("data", []) or []:
        parsed = _parse_price_entry(entry)
        if parsed:
            prices.append(parsed)
    return prices


def _get_inapp_purchase_snapshot(inapp_id: str) -> Dict[str, Any]:
    try:
        response = _request(
            "GET",
            f"/inAppPurchases/{inapp_id}",
            params={"include": "inAppPurchaseLocalizations,inAppPurchasePrices"},
        )
    except RuntimeError as exc:
        if not _is_parameter_error(exc):
            raise
        logger.info(
            "Apple API rejected include when fetching in-app purchase %s; "
            "falling back to compatibility mode.",
            inapp_id,
        )
    else:
        data = response.get("data", {})
        included = response.get("included", [])
        localization_map = _index_included(
            included, "inAppPurchaseLocalizations"
        )
        prices_map = _index_included(included, "inAppPurchasePrices")
        result = _canonicalize_record(data, localization_map, prices_map)
        if result:
            return result

    base = _request("GET", f"/inAppPurchases/{inapp_id}")
    record = _canonicalize_record(base.get("data", {}), {}, {})
    record["localizations"] = _fetch_inapp_localizations(inapp_id)
    record["prices"] = _fetch_inapp_prices(inapp_id)
    return record


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
    existing_entries = _list_localization_entries(inapp_id)
    existing_by_locale = {
        entry.get("attributes", {}).get("locale"): entry
        for entry in existing_entries
        if isinstance(entry.get("attributes"), dict)
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

    refreshed = _list_localization_entries(inapp_id)
    locale_ids: Dict[str, str] = {}
    for entry in refreshed:
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
    try:
        response = _request("GET", "/inAppPurchasePricePoints", params=params)
    except RuntimeError as exc:
        if _is_forbidden_error(exc):
            raise AppleStorePermissionError(
                "Apple API 키에 가격 포인트를 조회할 권한이 없습니다."
            ) from exc
        raise
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
    try:
        existing_prices = _request(
            "GET", f"/inAppPurchases/{inapp_id}/prices"
        ).get("data", [])
    except RuntimeError as exc:
        if (
            _is_path_error(exc)
            or _is_forbidden_noop_error(exc)
            or _is_forbidden_error(exc)
        ):
            logger.warning(
                "Apple API denied price schedule lookup for %s; proceeding without "
                "removing existing prices.",
                inapp_id,
            )
            existing_prices = []
        else:
            raise
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

    return _get_inapp_purchase_snapshot(inapp_id)


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

    return _get_inapp_purchase_snapshot(inapp_id)


def delete_inapp_purchase(inapp_id: str) -> None:
    _request("DELETE", f"/inAppPurchases/{inapp_id}")


def list_price_tiers(territory: str = "KOR") -> List[Dict[str, Any]]:
    def _tier_sort_key(value: str) -> tuple[int, str]:
        digits = "".join(ch for ch in value if ch.isdigit())
        return (int(digits) if digits else 0, value)

    def _collect_from_response(
        tiers: Dict[str, Dict[str, Any]], response: Dict[str, Any]
    ) -> None:
        for entry in response.get("data", []) or []:
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

    tiers: Dict[str, Dict[str, Any]] = {}
    cursor: Optional[str] = None

    permission_message = (
        "Apple API 키에 인앱 가격 정보를 조회할 권한이 없습니다. "
        "App Store Connect에서 API 키에 App Manager 또는 Finance 역할이 포함되어 있는지 확인해 주세요."
    )

    try:
        while True:
            params = {
                "filter[territory]": territory,
                "page[limit]": 200,
            }
            if cursor:
                params["page[cursor]"] = cursor
            response = _request(
                "GET", "/inAppPurchasePricePoints", params=params
            )
            _collect_from_response(tiers, response)
            cursor = _extract_cursor(response.get("links", {}).get("next"))
            if not cursor:
                break
    except RuntimeError as exc:
        if _is_forbidden_error(exc):
            if _is_forbidden_noop_error(exc):
                logger.warning(
                    "Apple API rejected unrestricted price point listing; falling back to "
                    "tier enumeration."
                )
            else:
                raise AppleStorePermissionError(permission_message) from exc
        else:
            raise
        tiers.clear()
        chunk_size = 25
        for index in range(0, len(_PRICE_TIER_GUESS_RANGE), chunk_size):
            chunk = _PRICE_TIER_GUESS_RANGE[index : index + chunk_size]
            params = {
                "filter[territory]": territory,
                "filter[priceTier]": ",".join(chunk),
                "page[limit]": 200,
            }
            try:
                response = _request(
                    "GET", "/inAppPurchasePricePoints", params=params
                )
            except RuntimeError as inner_exc:
                if _is_parameter_error(inner_exc):
                    logger.debug(
                        "Ignoring parameter error when probing price tiers chunk %s",
                        chunk,
                    )
                    continue
                if _is_forbidden_error(inner_exc):
                    raise AppleStorePermissionError(permission_message) from inner_exc
                raise
            _collect_from_response(tiers, response)

    return [tiers[key] for key in sorted(tiers.keys(), key=_tier_sort_key)]
