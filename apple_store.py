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
import concurrent.futures
import uuid
import sys
import threading
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import requests

# Ensure the exception placeholder exists at module scope (Py3.11+ clears except vars)
_jwt_import_error = None  # type: ignore[assignment]
try:  # pragma: no cover - import-time environment check
    import jwt as _jwt_module
except ImportError as _exc:  # pragma: no cover - handled lazily
    _jwt_module = None  # type: ignore[assignment]
    _jwt_import_error = _exc  # capture for later error chaining
else:  # pragma: no cover - exercised when dependency is available
    _jwt_import_error = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_APPLE_API_BASE = os.getenv(
    "APP_STORE_API_BASE_URL", "https://api.appstoreconnect.apple.com/v1"
)
_PRICE_POINTS_CACHE_TTL = int(os.getenv("APP_STORE_PRICE_POINT_CACHE_TTL", "1800"))
_APPLE_API_TIMEOUT = int(os.getenv("APPLE_API_TIMEOUT", "60"))  # Increased default timeout

_TOKEN_LOCK = threading.Lock()
_TOKEN_CACHE: Optional[Tuple[str, int]] = None

# Fallback token cache for price points API (with App Manager/Finance permissions)
_FALLBACK_TOKEN_CACHE: Optional[Tuple[str, int]] = None
_USE_FALLBACK_FOR_PRICE_POINTS = False

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
_INAPP_LIST_SUPPORTS_V2_ENDPOINT = True
_INAPP_PRICE_RELATIONSHIP_AVAILABLE = True
_PRICE_TIER_ACCESS_AVAILABLE = True
# appPricePoints endpoint uses simple limit/cursor params, not page[limit]
_PRICE_POINTS_SUPPORTS_PAGE_PARAMS = False

_INAPP_V2_ID_CACHE: Dict[str, Optional[str]] = {}
_INAPP_V2_ID_BY_PRODUCT_ID: Dict[str, Optional[str]] = {}

_LOCALIZATION_LIST_STRATEGY: Optional[str] = None


_PRICE_TIER_GUESS_RANGE = tuple(str(value) for value in range(0, 201))

_PRICE_POINT_REFERENCE_IAP_ID: Optional[str] = None
_PRICE_POINT_REFERENCE_LOCK = threading.Lock()
_PRICE_TIER_CACHE_TTL = int(os.getenv("APPLE_PRICE_TIER_CACHE_TTL", "3600"))
_PRICE_TIER_CACHE: Dict[str, "_PriceTierCacheEntry"] = {}
_PRICE_TIER_CACHE_LOCK = threading.Lock()


_ALL_TERRITORIES_CACHE: Optional[List[str]] = None
_ALL_TERRITORIES_LOCK = threading.Lock()


def _load_all_territories() -> List[str]:
    global _ALL_TERRITORIES_CACHE

    with _ALL_TERRITORIES_LOCK:
        if _ALL_TERRITORIES_CACHE is not None:
            return list(_ALL_TERRITORIES_CACHE)

    try:
        territories: List[str] = []
        cursor: Optional[str] = None

        while True:
            params: Dict[str, Any] = {"limit": 200}
            if cursor:
                params["page[cursor]"] = cursor

            response = _request("GET", "/v1/territories", params=params)
            for entry in response.get("data", []) or []:
                territory_id = (entry.get("id") or "").strip().upper()
                if territory_id and territory_id not in territories:
                    territories.append(territory_id)

            cursor = _extract_cursor(response.get("links", {}).get("next"))
            if not cursor:
                break

        if not territories:
            raise RuntimeError("Apple API에서 테리토리 목록을 반환하지 않았습니다.")

        with _ALL_TERRITORIES_LOCK:
            _ALL_TERRITORIES_CACHE = list(territories)

        return territories
    except Exception as exc:
        logger.warning(
            "전체 테리토리 목록을 불러오지 못했습니다. 고정 테리토리 목록으로 대체합니다: %s",
            exc,
        )
        with _ALL_TERRITORIES_LOCK:
            _ALL_TERRITORIES_CACHE = list(_FIXED_PRICE_TERRITORIES)
        return list(_FIXED_PRICE_TERRITORIES)


def _collect_manual_price_territories(base_territory: str) -> List[str]:
    territories: List[str] = []
    normalized_base = (base_territory or "").strip().upper()
    if normalized_base:
        territories.append(normalized_base)
    for fixed_territory in _FIXED_PRICE_TERRITORIES:
        normalized = fixed_territory.strip().upper()
        if normalized and normalized not in territories:
            territories.append(normalized)
    return territories


def _collect_availability_territories(base_territory: str) -> List[str]:
    territories: List[str] = []
    normalized_base = (base_territory or "").strip().upper()
    if normalized_base:
        territories.append(normalized_base)

    for territory in _load_all_territories():
        normalized = territory.strip().upper()
        if normalized and normalized not in territories:
            territories.append(normalized)
    return territories


def _parse_territory_list(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    territories: List[str] = []
    for token in re.split(r"[,\s]+", value):
        candidate = token.strip().upper()
        if not candidate:
            continue
        if candidate not in territories:
            territories.append(candidate)
    return territories


_DEFAULT_FIXED_PRICE_TERRITORIES = ("TWN", "MAC", "HKG")
_FIXED_PRICE_TERRITORIES: Tuple[str, ...] = tuple(
    _parse_territory_list(os.getenv("APPLE_FIXED_PRICE_TERRITORIES"))
    or _DEFAULT_FIXED_PRICE_TERRITORIES
)


def get_fixed_price_territories() -> Tuple[str, ...]:
    return _FIXED_PRICE_TERRITORIES


def _normalize_error_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key in ("id", "status", "code", "title", "detail"):
        value = entry.get(key)
        if value is None:
            continue
        if isinstance(value, (str, int)):
            normalized[key] = str(value)
        else:
            normalized[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return normalized


def _summarize_api_errors(errors: Iterable[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for entry in errors:
        if not isinstance(entry, dict):
            continue
        normalized = _normalize_error_entry(entry)
        code = normalized.get("code") or normalized.get("status")
        detail = normalized.get("detail") or normalized.get("title")
        if code or detail:
            snippet = " ".join(
                filter(None, [f"[{code}]" if code else "", detail])
            )
            parts.append(snippet)
        else:
            remaining = {
                key: value
                for key, value in normalized.items()
                if key not in {"code", "status", "detail", "title"}
            }
            if remaining:
                parts.append(json.dumps(remaining, ensure_ascii=False, sort_keys=True))
    return "; ".join(parts)


def _compose_permission_error_message(
    exc: Optional[Exception], default_message: str
) -> str:
    if isinstance(exc, AppleStoreApiError):
        detail = _summarize_api_errors(exc.errors)
        if detail:
            return f"{default_message} 원본 오류: {detail}"
        if exc.body_text:
            return f"{default_message} 원본 오류: {exc.body_text}"
    if exc:
        message = str(exc).strip()
        if message and message != default_message:
            return f"{default_message} 원본 오류: {message}"
    return default_message


class AppleStoreConfigError(RuntimeError):
    """Raised when required Apple configuration is missing."""


class AppleStorePermissionError(AppleStoreConfigError):
    """Raised when the App Store Connect API denies an operation."""


class AppleStoreApiError(RuntimeError):
    """Represents an error response returned by the Apple API."""

    def __init__(
        self,
        status_code: int,
        body_text: str,
        errors: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.status_code = status_code
        self.body_text = body_text
        self.errors = errors or []
        message = self._build_message()
        super().__init__(message)

    def _build_message(self) -> str:
        if self.errors:
            summary = _summarize_api_errors(self.errors)
            if summary:
                return f"Apple API 오류 {self.status_code}: {summary}"
        if self.body_text:
            return f"Apple API 오류 {self.status_code}: {self.body_text}"
        return f"Apple API 오류 {self.status_code}"


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

    logger.debug(
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


def _get_fallback_env(name: str) -> Optional[str]:
    """Get fallback environment variable value (e.g., FALLBACK_APP_STORE_KEY_ID)."""
    fallback_key = f"FALLBACK_{name}"
    value = os.getenv(fallback_key)
    if value:
        return value.strip()
    return None


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


def _load_private_key(use_fallback: bool = False) -> str:
    key_name = "FALLBACK_APP_STORE_PRIVATE_KEY_PATH" if use_fallback else "APP_STORE_PRIVATE_KEY_PATH"
    
    if use_fallback:
        path = _get_fallback_env("APP_STORE_PRIVATE_KEY_PATH")
        if not path:
            raise AppleStoreConfigError(
                f"Fallback key requested but {key_name} not configured"
            )
    else:
        path = _require_env("APP_STORE_PRIVATE_KEY_PATH")
    
    try:
        with open(path, "r", encoding="utf-8") as fp:
            contents = fp.read()
    except OSError as exc:  # pragma: no cover - depends on environment
        raise AppleStoreConfigError(
            f"{key_name}에서 키를 읽을 수 없습니다: {exc}"
        ) from exc

    contents = contents.lstrip("\ufeff").strip()
    if "-----BEGIN" not in contents or "PRIVATE KEY-----" not in contents:
        raise AppleStoreConfigError(
            "Apple API 비공개 키 파일 형식이 올바르지 않습니다. App Store Connect에서 내려받은 .p8 파일인지 확인해 주세요."
        )
    return contents + ("\n" if not contents.endswith("\n") else "")


def _generate_token(use_fallback: bool = False) -> str:
    """Generate JWT token. If use_fallback=True, uses fallback credentials for price points."""
    global _TOKEN_CACHE, _FALLBACK_TOKEN_CACHE
    
    if use_fallback:
        # Use fallback credentials (App Manager/Finance role for price points)
        issuer_id_key = "FALLBACK_APP_STORE_ISSUER_ID"
        key_id_key = "FALLBACK_APP_STORE_KEY_ID"
        
        fallback_issuer = _get_fallback_env("APP_STORE_ISSUER_ID")
        fallback_key_id = _get_fallback_env("APP_STORE_KEY_ID")
        
        if not fallback_issuer or not fallback_key_id:
            raise AppleStoreConfigError(
                "Fallback credentials not configured. Set FALLBACK_APP_STORE_ISSUER_ID, "
                "FALLBACK_APP_STORE_KEY_ID, and FALLBACK_APP_STORE_PRIVATE_KEY_PATH"
            )
        
        # Validate fallback credentials
        if not _UUID_RE.match(fallback_issuer):
            raise AppleStoreConfigError(f"{issuer_id_key} 값이 올바른 UUID 형식이 아닙니다.")
        if not _KEY_ID_RE.match(fallback_key_id.upper()):
            raise AppleStoreConfigError(f"{key_id_key} 값이 올바른 Key ID 형식이 아닙니다.")
        
        issuer_id = fallback_issuer
        key_id = fallback_key_id.upper()
        private_key = _load_private_key(use_fallback=True)
        cache_ref = _FALLBACK_TOKEN_CACHE
    else:
        # Use primary credentials
        issuer_id = _require_uuid_env("APP_STORE_ISSUER_ID")
        key_id = _require_key_id_env("APP_STORE_KEY_ID")
        private_key = _load_private_key(use_fallback=False)
        cache_ref = _TOKEN_CACHE

    now = int(time.time())
    with _TOKEN_LOCK:
        cached = cache_ref
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
        
        # Update the appropriate cache
        if use_fallback:
            _FALLBACK_TOKEN_CACHE = (token, expires_at)
        else:
            _TOKEN_CACHE = (token, expires_at)
        
        return token


def generate_jwt(*, force_refresh: bool = False) -> str:
    """Return a JWT for the App Store Connect API.

    Parameters
    ----------
    force_refresh:
        When ``True`` the cached token (if any) is discarded so that a new token
        is generated. This can be useful for scripts that always want a fresh
        token.
    """

    if force_refresh:
        _invalidate_token_cache()
    return _generate_token()


def _auth_headers(use_fallback: bool = False) -> Dict[str, str]:
    """Generate authorization headers. If use_fallback=True, uses fallback credentials."""
    return {
        "Authorization": f"Bearer {_generate_token(use_fallback=use_fallback)}",
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
    use_fallback: bool = False,
) -> Dict[str, Any]:
    if not path.startswith("/"):
        path = "/" + path
    base_url = _APPLE_API_BASE.rstrip("/")
    
    # Handle v2 paths: strip /v1 from base_url if needed
    if path.startswith("/v2/") and base_url.endswith("/v1"):
        base_url = base_url[: -len("/v1")] or base_url
    
    # Handle v1 paths: if path already has /v1/, and base_url ends with /v1,
    # strip /v1 from base_url to avoid double /v1/v1
    if path.startswith("/v1/") and base_url.endswith("/v1"):
        base_url = base_url[: -len("/v1")] or base_url
    
    url = base_url + path
    logger.debug("Apple API Request %s %s", method, url)
    if logger.isEnabledFor(logging.DEBUG) and params:
        logger.debug("  Params: %s", params)
    
    # Retry logic for timeout errors
    max_retries = 3
    retry_delay = 2
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            response = requests.request(
                method,
                url,
                headers=_auth_headers(use_fallback=use_fallback),
                params=params,
                json=json,
                timeout=_APPLE_API_TIMEOUT,
            )
            break  # Success, exit retry loop
        except requests.exceptions.Timeout as exc:
            last_exception = exc
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logger.warning(
                    "Timeout on %s %s (attempt %d/%d), retrying in %ds...",
                    method, url, attempt + 1, max_retries, wait_time
                )
                time.sleep(wait_time)
                continue
            else:
                logger.error(
                    "Timeout on %s %s after %d attempts, giving up",
                    method, url, max_retries
                )
                raise
        except requests.exceptions.RequestException as exc:
            # Other request errors (connection errors, etc.) - don't retry
            logger.error("Request error on %s %s: %s", method, url, exc)
            raise
    
    if last_exception and 'response' not in locals():
        raise last_exception
    if response.status_code >= 400:
        body_text = response.text.strip()
        errors: List[Dict[str, Any]] = []
        if body_text:
            try:
                payload = response.json()
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                raw_errors = payload.get("errors")
                if isinstance(raw_errors, list):
                    errors = [
                        entry
                        for entry in raw_errors
                        if isinstance(entry, dict)
                    ]
        
        # Determine log level based on error type
        log_level = logging.ERROR
        if response.status_code == 403 and errors:
            # Check if it's a "no allowed operations" error for price points
            for error in errors:
                error_detail = (error.get("detail") or "").lower()
                if "inapppurchasepricepoints" in error_detail and "no allowed operations" in error_detail:
                    log_level = logging.INFO  # Not an error, just missing permission
                    break
        
        summary = _summarize_api_errors(errors) if errors else body_text or ""
        logger.log(
            log_level,
            "Apple API error %s: %s | URL: %s",
            response.status_code,
            summary or "No response body",
            url,
            extra={
                "status": response.status_code,
                "body": body_text,
                "errors": errors,
                "url": url,
            },
        )
        if response.status_code == 401:
            _invalidate_token_cache()
            raise AppleStoreConfigError(
                _format_authorization_error(body_text)
            )
        raise AppleStoreApiError(response.status_code, body_text, errors)
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


def _list_inapp_purchase_identifiers_v2(
    *,
    cursor: Optional[str] = None,
    limit: int = 200,
    count_only: bool = False,
) -> Tuple[List[Tuple[str, str]], Optional[str], Optional[int]]:
    # Try v1-style endpoint first, v2 might not be available for all apps
    endpoints = [
        f"/apps/{_get_app_id()}/inAppPurchasesV2",
        f"/v2/apps/{_get_app_id()}/inAppPurchasesV2",
    ]
    
    params: Dict[str, Any] = {}
    if cursor:
        params["cursor"] = cursor

    if limit:
        if limit < 0:
            limit = 0
        if limit:
            params["limit"] = min(limit, 200)

    response = None
    for endpoint in endpoints:
        try:
            logger.debug("Trying listing endpoint: %s", endpoint)
            response = _request("GET", endpoint, params=params or None)
            logger.debug("Successfully fetched from endpoint: %s", endpoint)
            break
        except RuntimeError as exc:
            if _is_path_error(exc) or _is_not_found_error(exc):
                if endpoint == endpoints[0]:
                    logger.debug("v1-style endpoint failed, trying v2 endpoint")
                    continue
                logger.warning("All listing endpoints failed, will fallback to legacy mode")
                raise
    
    if response is None:
        raise RuntimeError("Failed to fetch in-app purchase list from any endpoint")

    # Extract total count if available
    total_count = None
    meta = response.get("meta", {})
    if isinstance(meta, dict):
        pagination = meta.get("paging", {})
        if isinstance(pagination, dict):
            total = pagination.get("total")
            if isinstance(total, int):
                total_count = total

    identifiers: List[Tuple[str, str]] = []
    for entry in response.get("data", []) or []:
        if not isinstance(entry, dict):
            continue
        inapp_id = entry.get("id")
        resource_type = entry.get("type") or "inAppPurchases"
        if isinstance(inapp_id, str):
            identifiers.append((inapp_id, resource_type))

    next_cursor = _extract_cursor(response.get("links", {}).get("next"))
    
    # Don't return next_cursor if we have no identifiers to avoid infinite loops
    # This can happen when:
    # 1. First request returns 0 items (empty app)
    # 2. Subsequent pagination returns 0 items (API bug)
    if not identifiers:
        logger.debug("No identifiers returned, clearing next_cursor to prevent infinite loop")
        next_cursor = None
    
    return identifiers, next_cursor, total_count


def _get_app_id() -> str:
    return _require_env("APP_STORE_APP_ID")


def setup_interrupt_handler() -> None:
    """Placeholder: interrupt handler is no longer used."""
    pass


def check_interrupted() -> bool:
    """Placeholder: interrupt check is no longer used."""
    return False


def _is_parameter_error(exc: Exception) -> bool:
    if isinstance(exc, AppleStoreApiError):
        if exc.status_code == 400:
            return True
        for entry in exc.errors:
            code = (entry.get("code") or entry.get("status") or "").upper()
            if "PARAMETER" in code:
                return True
            detail = (entry.get("detail") or entry.get("title") or "").lower()
            if any(
                marker in detail
                for marker in (
                    "not a valid relationship name",
                    "not a valid field name",
                    "not be used with this request",
                )
            ):
                return True
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
    if "PATH_ERROR" in message or "The URL path is not valid" in message:
        return True
    return "The path provided does not match a defined resource type" in message


def _is_not_found_error(exc: Exception) -> bool:
    message = str(exc)
    if not message:
        return False
    if "NOT_FOUND" in message:
        return True
    return "There is no resource of type" in message


def _should_disable_v2_detail_lookup(exc: Exception) -> bool:
    message = str(exc)
    if not message:
        return False
    markers = (
        "The path provided does not match a defined resource type",
        "There is no resource of type 'inAppPurchases' with id",
        "There is no resource of type 'inAppPurchasesV2' with id",
    )
    return any(marker in message for marker in markers)


def _is_forbidden_noop_error(exc: Exception) -> bool:
    if isinstance(exc, AppleStoreApiError):
        if exc.status_code == 403:
            for entry in exc.errors:
                detail = (entry.get("detail") or entry.get("title") or "").lower()
                if "no allowed operations" in detail:
                    return True
        return False
    message = str(exc)
    if not message:
        return False
    if "FORBIDDEN_ERROR" not in message:
        return False
    return "no allowed operations" in message.lower()


def _is_forbidden_error(exc: Exception) -> bool:
    if isinstance(exc, AppleStoreApiError):
        if exc.status_code == 403:
            return True
        for entry in exc.errors:
            code = (entry.get("code") or entry.get("status") or "").upper()
            if "FORBIDDEN" in code:
                return True
        return False
    message = str(exc)
    if not message:
        return False
    return "FORBIDDEN_ERROR" in message


def _is_rate_limit_error(exc: Exception) -> bool:
    if isinstance(exc, AppleStoreApiError):
        if exc.status_code == 429:
            return True
        for entry in exc.errors:
            code = (entry.get("code") or entry.get("status") or "").upper()
            if "RATE_LIMIT" in code or "TOO_MANY" in code:
                return True
        return False
    message = str(exc).upper()
    if not message:
        return False
    return "RATE_LIMIT" in message or "TOO_MANY" in message


def list_inapp_purchases(
    cursor: Optional[str] = None,
    limit: int = 200,
    *,
    include_relationships: bool = True,
    cumulative_fetched: int = 0,
    summary_only: bool = False,
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[int]]:
    global _INAPP_LIST_SUPPORTS_EXTENDED_PARAMS, _INAPP_LIST_SUPPORTS_LIMIT_PARAM
    global _INAPP_LIST_SUPPORTS_V2_ENDPOINT

    if _INAPP_LIST_SUPPORTS_V2_ENDPOINT:
        try:
            identifiers, next_cursor, total_count = _list_inapp_purchase_identifiers_v2(
                cursor=cursor, limit=limit
            )
        except RuntimeError as exc:
            if _is_parameter_error(exc) or _is_path_error(exc):
                logger.info(
                    "Apple API rejected inAppPurchasesV2 listing; falling back to legacy mode."
                )
                _INAPP_LIST_SUPPORTS_V2_ENDPOINT = False
            else:
                raise
        else:
            logger.info("Listing %d in-app purchases", len(identifiers))
            
            # If we have 0 identifiers, return empty result immediately
            if len(identifiers) == 0:
                logger.info("No IAPs to fetch in this batch, returning empty result")
                return [], next_cursor, total_count
            
            # Fetch IAPs in parallel for better performance
            # Process in batches to avoid rate limiting
            max_workers = min(3, len(identifiers))  # Further reduce concurrent requests
            batch_size = 30  # Smaller batches to avoid rate limits
            items: List[Dict[str, Any]] = []
            failed_items: List[Tuple[Tuple[str, str], int]] = []  # Store failed IAPs for retry
            had_lookup_failure = False
            had_success = False
            
            def fetch_iap_with_retry(
                identifier: Tuple[str, str],
                idx: int,
                max_retries: int = 5,
            ) -> Optional[Dict[str, Any]]:
                inapp_id, resource_type = identifier
                for attempt in range(max_retries):
                    try:
                        record = _get_inapp_purchase_snapshot(
                            inapp_id,
                            include_relationships=include_relationships,
                            summary_only=summary_only,
                        )
                        record.setdefault("resourceType", resource_type)
                        if not include_relationships:
                            record.pop("localizations", None)
                            record.pop("prices", None)
                        logger.debug("Successfully fetched IAP %d/%d: id=%s", idx, len(identifiers), inapp_id)
                        return record
                    except RuntimeError as exc:
                        if _is_rate_limit_error(exc):
                            if attempt < max_retries - 1:
                                wait_time = min(2 ** (attempt + 1), 30)  # Cap at 30 seconds
                                logger.info(
                                    "Rate limited on IAP %d/%d (id=%s), retry %d/%d in %ds",
                                    idx, len(identifiers), inapp_id, attempt + 1, max_retries, wait_time
                                )
                                time.sleep(wait_time)
                                continue
                            logger.warning(
                                "Rate limited on IAP %d/%d (id=%s) after %d retries - will retry later",
                                idx, len(identifiers), inapp_id, max_retries
                            )
                            return None
                        
                        if not (_is_not_found_error(exc) or _is_forbidden_error(exc) or _is_path_error(exc)):
                            # Fatal error - re-raise
                            logger.error("Fatal error fetching IAP %d/%d (id=%s): %s", idx, len(identifiers), inapp_id, exc)
                            raise
                        # Non-fatal error - log and return None
                        logger.debug("Skipping IAP %d/%d (id=%s): %s", idx, len(identifiers), inapp_id, exc)
                        return None
                    except Exception as exc:
                        if attempt == max_retries - 1:
                            logger.error(
                                "Unexpected error fetching IAP %d/%d (id=%s): %s",
                                idx,
                                len(identifiers),
                                inapp_id,
                                exc,
                                exc_info=True
                            )
                        return None
                return None
            
            def fetch_iap(identifier: Tuple[str, str], idx: int) -> Optional[Dict[str, Any]]:
                return fetch_iap_with_retry(identifier, idx)
            
            # Process in batches to avoid overwhelming the API
            for batch_start in range(0, len(identifiers), batch_size):
                # Check for interrupt
                if check_interrupted():
                    logger.info("Fetching interrupted by user")
                    break
                
                batch_end = min(batch_start + batch_size, len(identifiers))
                batch = list(enumerate(identifiers[batch_start:batch_end], batch_start + 1))
                
                # Calculate remaining items across all pages
                items_fetched_so_far = cumulative_fetched + batch_end
                if total_count:
                    remaining = total_count - items_fetched_so_far
                    logger.info(
                        "Processing batch: %d-%d of %d (%d/%d fetched, %d remaining)", 
                        batch_start + 1, batch_end, len(identifiers), items_fetched_so_far, total_count, remaining
                    )
                else:
                    logger.info(
                        "Processing batch: %d-%d of %d", 
                        batch_start + 1, batch_end, len(identifiers)
                    )
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_idx = {
                        executor.submit(fetch_iap, identifier, idx): idx
                        for idx, identifier in batch
                    }
                    
                    for future in concurrent.futures.as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            result = future.result()
                            if result:
                                items.append(result)
                                had_success = True
                            else:
                                had_lookup_failure = True
                                # Store failed item for retry
                                failed_items.append((identifiers[idx - 1], idx))
                        except Exception as exc:
                            logger.error("Failed to process IAP %d: %s", idx, exc)
                            had_lookup_failure = True
                            failed_items.append((identifiers[idx - 1], idx))
                
                # Add a longer delay between batches to avoid rate limiting
                if batch_end < len(identifiers):
                    time.sleep(1.0)
            
            # Retry failed items with longer delays
            if failed_items and not check_interrupted():
                logger.info("Retrying %d failed IAPs...", len(failed_items))
                time.sleep(2)  # Wait before retry
                
                retry_success = 0
                for identifier, idx in failed_items:
                    if check_interrupted():
                        logger.info("Retry interrupted by user")
                        break
                    try:
                        result = fetch_iap_with_retry(identifier, idx, max_retries=10)
                        if result:
                            items.append(result)
                            retry_success += 1
                            logger.debug("Successfully retried IAP %d: id=%s", idx, identifier[0])
                            time.sleep(0.5)  # Delay between retries
                        else:
                            logger.warning("Failed to retry IAP %d: id=%s", idx, identifier[0])
                    except Exception as exc:
                        logger.error("Error retrying IAP %d: id=%s, error=%s", idx, identifier[0], exc)
                
                if retry_success > 0:
                    logger.info("Retried %d/%d failed IAPs successfully", retry_success, len(failed_items))
            
            # Log summary for this page
            fetched_count = len(items)
            total_to_fetch = len(identifiers)
            if total_count:
                total_fetched = cumulative_fetched + fetched_count
                remaining = total_count - total_fetched
                logger.info(
                    "Page complete: %d/%d IAPs fetched from this page (%d/%d total, %d remaining)", 
                    fetched_count, total_to_fetch, total_fetched, total_count, remaining
                )
            else:
                logger.info(
                    "Page complete: %d/%d IAPs fetched from this page", 
                    fetched_count, total_to_fetch
                )
            
            if had_lookup_failure:
                logger.warning(
                    "Completed with %d items missing due to errors", 
                    total_to_fetch - fetched_count
                )
            
            # Only disable V2 and retry if we had ONLY failures (no successes at all)
            # But still return partial results if we had some successes
            if had_lookup_failure and not had_success and _INAPP_LIST_SUPPORTS_V2_ENDPOINT:
                logger.info(
                    "Disabling inAppPurchasesV2 detail lookups after all lookups failed."
                )
                _INAPP_LIST_SUPPORTS_V2_ENDPOINT = False
                items, next_cursor, _ = list_inapp_purchases(
                    cursor=cursor,
                    limit=limit,
                    include_relationships=include_relationships,
                    summary_only=summary_only,
                )
                return items, next_cursor, total_count
            
            # Return partial results if we had some successes
            return items, next_cursor, total_count

    endpoint = f"/apps/{_get_app_id()}/inAppPurchases"

    if _INAPP_LIST_SUPPORTS_EXTENDED_PARAMS:
        params: Dict[str, Any] = {
            "fields[inAppPurchases]": ",".join(
                [
                    "productId",
                    "referenceName",
                    "inAppPurchaseType",
                    "state",
                ]
            ),
        }
        if include_relationships:
            params["include"] = "inAppPurchaseLocalizations"
            params["fields[inAppPurchaseLocalizations]"] = "locale,name"
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
                item = _canonicalize_record(
                    record,
                    localization_map,
                    prices_map,
                    with_relationships=include_relationships,
                )
                resolved_id, resolved_type = _resolve_inapp_v2_identifier(
                    item.get("id"),
                    item.get("resourceType", "inAppPurchases"),
                    item.get("productId"),
                )
                if resolved_id:
                    item["id"] = resolved_id
                item["resourceType"] = resolved_type
                finalized = _finalize_inapp_record(
                    item,
                    summary_only=summary_only,
                )
                if not summary_only and not include_relationships:
                    finalized.pop("localizations", None)
                    finalized.pop("prices", None)
                items.append(finalized)

            next_cursor = _extract_cursor(response.get("links", {}).get("next"))
            return items, next_cursor, None  # No total count available in legacy mode

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
        item = _canonicalize_record(
            record, {}, {}, with_relationships=include_relationships
        )
        inapp_id, resource_type = _resolve_inapp_v2_identifier(
            item.get("id"),
            item.get("resourceType", "inAppPurchases"),
            item.get("productId"),
        )
        if inapp_id:
            item["id"] = inapp_id
        item["resourceType"] = resource_type
        if include_relationships:
            item["localizations"] = _fetch_inapp_localizations(
                inapp_id, resource_type
            )
        else:
            item.pop("localizations", None)
        if summary_only:
            item.pop("prices", None)
        else:
            item["prices"] = _fetch_inapp_prices(inapp_id, resource_type)
        finalized = _finalize_inapp_record(
            item,
            summary_only=summary_only,
        )
        items.append(finalized)

    next_cursor = _extract_cursor(response.get("links", {}).get("next"))
    return items, next_cursor, None  # No total count available in basic mode


def iterate_all_inapp_purchases(
    limit: int = 200,
    *,
    include_relationships: bool = True,
    max_items: Optional[int] = None,
    summary_only: bool = False,
) -> Iterable[Dict[str, Any]]:
    cursor: Optional[str] = None
    cumulative_count = 0
    page_num = 0
    
    while True:
        page_num += 1
        items, cursor, total_count = list_inapp_purchases(
            cursor=cursor,
            limit=limit,
            include_relationships=include_relationships,
            cumulative_fetched=cumulative_count,
            summary_only=summary_only,
        )
        
        # Apply max_items limit if specified (for debug mode)
        if max_items is not None and cumulative_count + len(items) > max_items:
            remaining = max_items - cumulative_count
            if remaining > 0:
                items = items[:remaining]
            else:
                items = []
        
        cumulative_count += len(items)
        
        # Yield items
        for item in items:
            yield item
        
        # Stop if no more cursor OR if we received 0 items (completed) OR if we hit max_items limit
        if not cursor or len(items) == 0 or (max_items is not None and cumulative_count >= max_items):
            if cumulative_count > 0:
                logger.info("Fetch complete: %d total items retrieved", cumulative_count)
            break


def get_all_inapp_purchases(
    *,
    include_relationships: bool = True,
    max_items: Optional[int] = None,
    summary_only: bool = False,
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    items = list(
        iterate_all_inapp_purchases(
            include_relationships=include_relationships,
            max_items=max_items,
            summary_only=summary_only,
        )
    )
    return items, None  # Total count not available in iterate mode


def get_inapp_purchase_ids_lightweight() -> Tuple[List[str], Optional[int]]:
    """
    Fetch only IAP IDs without full details for efficient change detection.
    Returns (list of product IDs, total count)
    """
    if _INAPP_LIST_SUPPORTS_V2_ENDPOINT:
        try:
            identifiers, next_cursor, total_count = _list_inapp_purchase_identifiers_v2(limit=200)
            
            # Fetch all pages of product IDs only (no details)
            all_ids = [product_id for product_id, _ in identifiers if product_id]
            
            while next_cursor:
                identifiers, next_cursor, _ = _list_inapp_purchase_identifiers_v2(
                    cursor=next_cursor, limit=200
                )
                all_ids.extend([product_id for product_id, _ in identifiers if product_id])
            
            logger.info("Fetched %d IAP IDs (lightweight check)", len(all_ids))
            return all_ids, total_count
        except RuntimeError as exc:
            if _is_parameter_error(exc) or _is_path_error(exc):
                logger.info("V2 endpoint not available for lightweight check, falling back")
            else:
                raise
    
    # Fallback: fetch minimal data
    items, _ = get_all_inapp_purchases(include_relationships=False)
    ids = [item.get("productId") or item.get("sku") or "" for item in items if item]
    return [id for id in ids if id], None


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


def _parse_price_point_entry(
    entry: Dict[str, Any],
    price_tier_map: Dict[str, Dict[str, Any]],
    territory_map: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Parse V2 price point entry with included priceTier and territory."""
    if not isinstance(entry, dict):
        return None
    
    relationships = entry.get("relationships") or {}
    
    # Get price tier from relationship
    price_tier_data = relationships.get("priceTier", {}).get("data") or {}
    price_tier_id = price_tier_data.get("id", "")
    price_tier_info = price_tier_map.get(price_tier_id, {})
    price_tier_attrs = price_tier_info.get("attributes") or {}
    
    # Get territory from relationship
    territory_data = relationships.get("territory", {}).get("data") or {}
    territory_id = territory_data.get("id", "")
    
    if not territory_id or not price_tier_id:
        return None
    
    return {
        "id": entry.get("id"),
        "currency": price_tier_attrs.get("currency"),
        "price": None,  # Not available in price point response
        "startDate": None,  # Not available in price point response
        "territory": territory_id,
        "priceTier": price_tier_attrs.get("priceTier"),
    }


def _extract_relationship_id(data: object) -> Optional[str]:
    if isinstance(data, dict):
        candidate = data.get("id")
        if isinstance(candidate, str):
            return candidate
    elif isinstance(data, list):
        for entry in data:
            candidate = _extract_relationship_id(entry)
            if candidate:
                return candidate
    return None


def _canonicalize_record(
    record: Dict[str, Any],
    localization_map: Dict[str, Dict[str, Any]],
    prices_map: Dict[str, Dict[str, Any]],
    *,
    with_relationships: bool = True,
) -> Dict[str, Any]:
    attributes = record.get("attributes") or {}
    relationships = record.get("relationships") or {}

    localizations: Dict[str, Dict[str, str]] = {}
    if with_relationships:
        localization_relationship = (
            relationships.get("inAppPurchaseLocalizations") or {}
        )
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
    if with_relationships:
        prices_relationship = relationships.get("inAppPurchasePrices") or {}
        for price_ref in prices_relationship.get("data", []) or []:
            price_id = price_ref.get("id")
            if not price_id:
                continue
            price_entry = prices_map.get(price_id) or price_ref
            parsed_price = _parse_price_entry(price_entry)
            if parsed_price:
                prices.append(parsed_price)

    resolved_id = record.get("id", "")
    resolved_resource_type = record.get("type", "inAppPurchases")
    for relationship_key in ("inAppPurchaseV2", "inAppPurchase"):
        relationship = relationships.get(relationship_key) or {}
        candidate = _extract_relationship_id(relationship.get("data"))
        if isinstance(candidate, str) and candidate.isdigit():
            resolved_id = candidate
            resolved_resource_type = "inAppPurchasesV2"
            break

    product_id = attributes.get("productId", "")
    return {
        "id": resolved_id,
        "resourceType": resolved_resource_type,
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


def _summarize_inapp_record(record: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(record, dict):
        return {}

    localizations = record.get("localizations") or {}
    preferred_name = ""
    for key in ("ko", "ko-KR", "ko_kr", "KOR"):
        loc_payload = localizations.get(key)
        if isinstance(loc_payload, dict):
            name = loc_payload.get("name")
            if name:
                preferred_name = str(name)
                break

    if not preferred_name:
        preferred_name = str(record.get("referenceName") or "")

    product_id = str(record.get("productId") or record.get("sku") or "")
    summary: Dict[str, Any] = {
        "id": record.get("id"),
        "resourceType": record.get("resourceType"),
        "productId": product_id,
        "sku": product_id,
        "referenceName": preferred_name,
        "type": record.get("type"),
        "state": record.get("state"),
    }

    krw_price = record.get("krwPrice")
    if isinstance(krw_price, dict) and krw_price:
        summary["krwPrice"] = krw_price
        price_tier = krw_price.get("priceTier")
        if price_tier:
            summary["priceTier"] = price_tier
    else:
        price_tier = record.get("priceTier")
        if price_tier:
            summary["priceTier"] = price_tier

    return summary


def _finalize_inapp_record(
    record: Dict[str, Any],
    *,
    summary_only: bool = False,
) -> Dict[str, Any]:
    if not isinstance(record, dict):
        return {}

    identifier = str(record.get("id") or "")
    price_info = _fetch_krw_price_with_retry(identifier) if identifier else None
    if price_info:
        record["krwPrice"] = price_info
        tier_value = price_info.get("priceTier")
        if tier_value:
            record["priceTier"] = tier_value

    if summary_only:
        return _summarize_inapp_record(record)

    return record


def _normalize_localization_entries(entries: object) -> List[Dict[str, Any]]:
    if isinstance(entries, list):
        return [entry for entry in entries if isinstance(entry, dict)]
    if isinstance(entries, dict):
        return [entries]
    return []


def _cache_v2_id(original_id: str, product_id: Optional[str], resolved_id: Optional[str]) -> None:
    _INAPP_V2_ID_CACHE[original_id] = resolved_id
    if product_id:
        _INAPP_V2_ID_BY_PRODUCT_ID[product_id] = resolved_id


def _resolve_inapp_v2_identifier(
    inapp_id: Optional[str],
    resource_type: str,
    product_id: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    if not inapp_id:
        return None, resource_type

    if inapp_id.isdigit():
        return inapp_id, "inAppPurchasesV2"

    if resource_type == "inAppPurchasesV2":
        return inapp_id, resource_type

    cached = _INAPP_V2_ID_CACHE.get(inapp_id)
    if cached is not None:
        return cached, "inAppPurchasesV2" if cached else resource_type

    if product_id:
        cached_product = _INAPP_V2_ID_BY_PRODUCT_ID.get(product_id)
        if cached_product is not None:
            _cache_v2_id(inapp_id, product_id, cached_product)
            return cached_product, "inAppPurchasesV2" if cached_product else resource_type

    relationship_paths = [
        f"/inAppPurchases/{inapp_id}/relationships/inAppPurchaseV2",
        f"/inAppPurchases/{inapp_id}/relationships/inAppPurchase",
    ]

    for rel_path in relationship_paths:
        try:
            response = _request("GET", rel_path)
        except RuntimeError as exc:
            if (
                _is_path_error(exc)
                or _is_parameter_error(exc)
                or _is_forbidden_noop_error(exc)
                or _is_forbidden_error(exc)
                or _is_not_found_error(exc)
            ):
                continue
            raise

        candidate = _extract_relationship_id(response.get("data"))
        if isinstance(candidate, str) and candidate.isdigit():
            _cache_v2_id(inapp_id, product_id, candidate)
            return candidate, "inAppPurchasesV2"

    if product_id:
        params = {"filter[productId]": product_id, "limit": "1"}
        endpoints = ["/v2/inAppPurchasesV2", "/inAppPurchasesV2"]
        
        for endpoint in endpoints:
            try:
                response = _request("GET", endpoint, params=params)
                break
            except RuntimeError as exc:
                if _is_path_error(exc) and endpoint == endpoints[0]:
                    # Try next endpoint
                    continue
                if (
                    _is_forbidden_noop_error(exc)
                    or _is_forbidden_error(exc)
                    or _is_not_found_error(exc)
                ):
                    _cache_v2_id(inapp_id, product_id, None)
                    return inapp_id, resource_type
                raise

        entries = response.get("data")
        if isinstance(entries, dict):
            entries = [entries]
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                candidate = entry.get("id")
                if isinstance(candidate, str) and candidate.isdigit():
                    _cache_v2_id(inapp_id, product_id, candidate)
                    return candidate, "inAppPurchasesV2"

    _cache_v2_id(inapp_id, product_id, None)
    return inapp_id, resource_type


def _load_localization_entries_via_filter(
    inapp_id: str, resource_type: str
) -> Optional[List[Dict[str, Any]]]:
    if resource_type != "inAppPurchasesV2" and inapp_id.isdigit():
        resource_type = "inAppPurchasesV2"

    filter_keys = ["inAppPurchase"]
    if resource_type == "inAppPurchasesV2":
        filter_keys = ["inAppPurchaseV2", "inAppPurchase"]

    for index, filter_key in enumerate(filter_keys):
        entries: List[Dict[str, Any]] = []
        supported = False
        cursor: Optional[str] = None

        while True:
            params: Dict[str, Any] = {
                f"filter[{filter_key}]": inapp_id,
                "limit": "200",
            }
            if cursor:
                params["cursor"] = cursor

            try:
                response = _request(
                    "GET", "/inAppPurchaseLocalizations", params=params
                )
            except RuntimeError as exc:
                has_alternate_filter = index + 1 < len(filter_keys)
                if _is_parameter_error(exc) and has_alternate_filter:
                    logger.info(
                        "Apple API rejected localization filter '%s' for %s; "
                        "retrying with alternate filter.",
                        filter_key,
                        inapp_id,
                    )
                    break
                if (
                    _is_parameter_error(exc)
                    or _is_forbidden_noop_error(exc)
                    or _is_forbidden_error(exc)
                    or _is_not_found_error(exc)
                ):
                    logger.warning(
                        "Apple API denied localization filter lookup for %s (filter=%s)",
                        inapp_id,
                        filter_key,
                    )
                    if supported:
                        return entries
                    return None
                raise

            entries.extend(_normalize_localization_entries(response.get("data")))
            supported = True

            cursor = _extract_cursor(response.get("links", {}).get("next"))
            if not cursor:
                return entries

        # Try next filter key if the current one was rejected

    return None


def _load_localization_entries_v2(
    inapp_id: str, resource_type: str
) -> Optional[List[Dict[str, Any]]]:
    entries: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    # For V2 IAPs (numeric IDs), use /v2/inAppPurchases/{id}/inAppPurchaseLocalizations
    # For V1 IAPs (UUIDs), use /v2/inAppPurchases/{id}/inAppPurchaseLocalizations
    candidates = [f"/v2/inAppPurchases/{inapp_id}/inAppPurchaseLocalizations"]

    for path in candidates:
        entries.clear()
        cursor = None

        while True:
            params: Dict[str, Any] = {"limit": "200"}
            if cursor:
                params["cursor"] = cursor

            try:
                response = _request("GET", path, params=params)
            except RuntimeError as exc:
                if (
                    _is_path_error(exc)
                    or _is_parameter_error(exc)
                    or _is_forbidden_noop_error(exc)
                    or _is_forbidden_error(exc)
                    or _is_not_found_error(exc)
                ):
                    logger.info(
                        "Apple API rejected v2 localization lookup for %s via %s",
                        inapp_id,
                        path,
                    )
                    break
                raise

            entries.extend(
                _normalize_localization_entries(response.get("data"))
            )

            cursor = _extract_cursor(response.get("links", {}).get("next"))
            if not cursor:
                return entries

    return None


def _load_localizations_via_relationship(
    inapp_id: str, resource_type: str
) -> Optional[List[Dict[str, Any]]]:
    if resource_type != "inAppPurchasesV2" and inapp_id.isdigit():
        resource_type = "inAppPurchasesV2"

    # For V2 IAPs (numeric IDs), use /v2/inAppPurchases/{id}/inAppPurchaseLocalizations
    # For V1 IAPs (UUIDs), use /inAppPurchases/{id}/inAppPurchaseLocalizations
    candidates = []
    if resource_type == "inAppPurchasesV2":
        candidates.append(f"/v2/inAppPurchases/{inapp_id}/inAppPurchaseLocalizations")
    candidates.append(f"/inAppPurchases/{inapp_id}/inAppPurchaseLocalizations")

    success = False
    for path in candidates:
        try:
            response = _request("GET", path)
        except RuntimeError as exc:
            if not (
                _is_path_error(exc)
                or _is_forbidden_noop_error(exc)
                or _is_forbidden_error(exc)
                or _is_not_found_error(exc)
            ):
                raise
            logger.info(
                "Apple API reported missing in-app localization relationship via %s; "
                "retrying with alternate lookup methods.",
                path,
            )
        else:
            success = True
            return _normalize_localization_entries(response.get("data"))

    if success:
        return []
    return None


def _list_localization_entries(
    inapp_id: str, resource_type: str
) -> List[Dict[str, Any]]:
    global _LOCALIZATION_LIST_STRATEGY

    strategies = []
    if _LOCALIZATION_LIST_STRATEGY:
        strategies.append(_LOCALIZATION_LIST_STRATEGY)
    strategies.extend(["relationship", "v2", "filter"])

    seen: set[str] = set()
    for strategy in strategies:
        if strategy in seen:
            continue
        seen.add(strategy)

        if strategy == "relationship":
            entries = _load_localizations_via_relationship(inapp_id, resource_type)
        elif strategy == "v2":
            entries = _load_localization_entries_v2(inapp_id, resource_type)
        elif strategy == "filter":
            entries = _load_localization_entries_via_filter(inapp_id, resource_type)
        else:
            continue

        if entries is None:
            if _LOCALIZATION_LIST_STRATEGY == strategy:
                _LOCALIZATION_LIST_STRATEGY = None
            continue

        _LOCALIZATION_LIST_STRATEGY = strategy
        return entries

    return []


def _fetch_inapp_localizations(
    inapp_id: Optional[str], resource_type: str = "inAppPurchases"
) -> Dict[str, Dict[str, str]]:
    if not inapp_id:
        return {}

    if resource_type != "inAppPurchasesV2" and inapp_id.isdigit():
        resource_type = "inAppPurchasesV2"

    result: Dict[str, Dict[str, str]] = {}
    for entry in _list_localization_entries(inapp_id, resource_type):
        attributes = entry.get("attributes") or {}
        locale = attributes.get("locale")
        if not locale:
            continue
        result[locale] = {
            "name": attributes.get("name", ""),
            "description": attributes.get("description", ""),
        }
    return result


def _fetch_inapp_prices(
    inapp_id: Optional[str], resource_type: str = "inAppPurchases"
) -> List[Dict[str, Any]]:
    global _INAPP_PRICE_RELATIONSHIP_AVAILABLE

    if not inapp_id:
        return []

    if resource_type != "inAppPurchasesV2" and inapp_id.isdigit():
        resource_type = "inAppPurchasesV2"

    if not _INAPP_PRICE_RELATIONSHIP_AVAILABLE:
        return []

    # Note: V2 pricePoints API has changed and no longer supports include parameters
    # We now rely on inAppPurchasePriceSchedules for price information
    # This function is kept for backward compatibility but may return empty list
    paths = []
    # Only try V1 endpoint for UUID-based IAPs
    if resource_type != "inAppPurchasesV2" and not inapp_id.isdigit():
        paths.append((f"/inAppPurchases/{inapp_id}/prices", False))

    for path, is_v2 in paths:
        try:
            response = _request("GET", path)
        except RuntimeError as exc:
            if (
                _is_path_error(exc)
                or _is_parameter_error(exc)
                or _is_forbidden_noop_error(exc)
                or _is_forbidden_error(exc)
                or _is_not_found_error(exc)
            ):
                continue
            # Don't raise, just log and continue
            logger.debug("Failed to fetch prices for IAP %s via %s: %s", inapp_id, path, exc)
            continue
        else:
            prices: List[Dict[str, Any]] = []
            # V1 response has prices directly
            for entry in response.get("data", []) or []:
                parsed = _parse_price_entry(entry)
                if parsed:
                    prices.append(parsed)
            return prices

    # Price relationship API is not reliable, disable it
    if _INAPP_PRICE_RELATIONSHIP_AVAILABLE:
        logger.debug(
            "Apple API price relationship not available for IAP %s; "
            "using price schedule API instead.",
            inapp_id,
        )
    _INAPP_PRICE_RELATIONSHIP_AVAILABLE = False
    return []


def _fetch_price_schedule_id(inapp_id: str) -> Optional[str]:
    if not inapp_id:
        return None

    path = (
        f"/v2/inAppPurchases/{inapp_id}/iapPriceSchedule"
        if inapp_id.isdigit()
        else f"/inAppPurchases/{inapp_id}/priceSchedule"
    )

    try:
        response = _request("GET", path)
    except RuntimeError as exc:
        if _is_not_found_error(exc) or _is_forbidden_error(exc):
            logger.debug("Price schedule lookup failed for IAP %s: %s", inapp_id, exc)
            return None
        raise

    data = response.get("data", {}) if isinstance(response, dict) else {}
    schedule_id = data.get("id")
    if not schedule_id:
        logger.debug("No price schedule ID found for IAP %s", inapp_id)
        return None
    return schedule_id


def _list_manual_prices(schedule_id: str, territory: Optional[str] = None) -> List[Dict[str, Any]]:
    if not schedule_id:
        return []
    params: Dict[str, Any] = {
        "include": "inAppPurchasePricePoint,territory",
    }
    if territory:
        params["filter[territory]"] = territory

    try:
        response = _request(
            "GET",
            f"/v1/inAppPurchasePriceSchedules/{schedule_id}/manualPrices",
            params=params,
        )
    except RuntimeError as exc:
        if _is_not_found_error(exc):
            logger.debug("Manual price list not found for schedule %s: %s", schedule_id, exc)
            return []
        raise

    data = response.get("data", []) if isinstance(response, dict) else []
    return data or []


def _clear_existing_manual_prices(inapp_id: str) -> None:
    schedule_id = _fetch_price_schedule_id(inapp_id)
    if not schedule_id:
        return

    manual_prices = _list_manual_prices(schedule_id)
    for entry in manual_prices:
        entry_id = entry.get("id")
        if not entry_id:
            continue
        try:
            _request("DELETE", f"/v1/inAppPurchasePrices/{entry_id}")
        except RuntimeError as exc:
            if _is_not_found_error(exc):
                continue
            logger.debug(
                "Failed to delete manual price %s for IAP %s: %s",
                entry_id,
                inapp_id,
                exc,
            )


def get_iap_price_krw(inapp_id: str) -> Optional[Dict[str, str]]:
    """Fetch KRW price for a given IAP via price schedule manual prices endpoint.

    Uses: GET /v1/inAppPurchasePriceSchedules/{SCHEDULE_ID}/manualPrices
          ?filter[territory]=KOR
          &include=inAppPurchasePricePoint,territory

    The current price is the manual price with startDate === null. The actual price
    values (customerPrice, proceeds) are retrieved from the related inAppPurchasePricePoint.

    Returns a dict like {"currency": "KRW", "customerPrice": str, "proceeds": str}
    or None if not found.
    """
    if not inapp_id:
        return None

    schedule_id = _fetch_price_schedule_id(inapp_id)
    if not schedule_id:
        logger.debug("No price schedule available for IAP %s", inapp_id)
        return None

    try:
        resp = _request(
            "GET",
            f"/v1/inAppPurchasePriceSchedules/{schedule_id}/manualPrices",
            params={
                "filter[territory]": "KOR",
                "include": "inAppPurchasePricePoint,territory",
            },
        )
    except RuntimeError as exc:
        logger.debug(
            "Failed to fetch KRW price for IAP %s via price schedules: %s",
            inapp_id,
            exc,
        )
        return None

    data = resp.get("data", [])
    if not data:
        logger.debug("No manual prices found for IAP %s (KOR territory)", inapp_id)
        return None

    # Find the current price (startDate === null)
    current_price_entry = None
    for entry in data:
        attrs = entry.get("attributes", {})
        start_date = attrs.get("startDate")
        if start_date is None:
            current_price_entry = entry
            break

    if not current_price_entry:
        logger.debug("No current price (startDate=null) found for IAP %s (KOR territory)", inapp_id)
        return None

    # Get the price point ID from the relationship
    relationships = current_price_entry.get("relationships", {})
    price_point_data = relationships.get("inAppPurchasePricePoint", {}).get("data", {})
    price_point_id = price_point_data.get("id", "") if isinstance(price_point_data, dict) else ""

    if not price_point_id:
        logger.debug("No price point relationship found for current price of IAP %s", inapp_id)
        return None

    # Find the price point in included resources
    included = resp.get("included", [])
    price_point_map = _index_included(included, "inAppPurchasePricePoints")
    territory_map = _index_included(included, "territories")

    price_point_obj = price_point_map.get(price_point_id)
    if not price_point_obj or not isinstance(price_point_obj, dict):
        logger.debug("Price point %s not found in included resources for IAP %s", price_point_id, inapp_id)
        return None

    # Get customerPrice and proceeds from price point attributes
    price_point_attrs = price_point_obj.get("attributes", {})
    customer_price = price_point_attrs.get("customerPrice")
    proceeds = price_point_attrs.get("proceeds")

    # Get currency from territory (default to KRW)
    territory_data = relationships.get("territory", {}).get("data", {})
    territory_id = territory_data.get("id", "") if isinstance(territory_data, dict) else ""

    currency = "KRW"
    if territory_id:
        terr_obj = territory_map.get(territory_id) or {}
        terr_attrs = terr_obj.get("attributes", {}) if isinstance(terr_obj, dict) else {}
        currency = terr_attrs.get("currency", "KRW")

    if customer_price is None:
        logger.debug("No customerPrice found in price point %s for IAP %s", price_point_id, inapp_id)
        return None

    # Convert to string (it might be a number)
    customer_price_str = str(customer_price)
    proceeds_str = str(proceeds or "")
    price_tier = price_point_attrs.get("priceTier")

    logger.info(
        "Fetched KRW price for IAP %s: %s %s (proceeds: %s, tier: %s)",
        inapp_id,
        currency,
        customer_price_str,
        proceeds_str,
        price_tier or "",
    )

    return {
        "currency": str(currency),
        "customerPrice": customer_price_str,
        "proceeds": proceeds_str,
        "priceTier": str(price_tier or ""),
    }


def _should_skip_krw_price_fetch() -> bool:
    raw = os.getenv("APPLE_SKIP_KRW_PRICE_FETCH", "")
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def _fetch_krw_price_with_retry(
    inapp_id: str, *, max_attempts: int = 3, initial_delay: float = 1.0
) -> Optional[Dict[str, str]]:
    if not inapp_id or not inapp_id.isdigit():
        return None
    if _should_skip_krw_price_fetch():
        logger.info(
            "Skipping KRW price fetch for IAP %s (APPLE_SKIP_KRW_PRICE_FETCH enabled)",
            inapp_id,
        )
        return None

    delay = initial_delay
    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            price = get_iap_price_krw(inapp_id)
        except Exception as exc:
            last_error = exc
            logger.warning(
                "KRW price fetch failed for IAP %s (attempt %d/%d): %s",
                inapp_id,
                attempt,
                max_attempts,
                exc,
            )
            price = None

        if price:
            return price

        if attempt < max_attempts:
            logger.info(
                "Retrying KRW price fetch for IAP %s (attempt %d/%d) after %.1fs",
                inapp_id,
                attempt + 1,
                max_attempts,
                delay,
            )
            time.sleep(min(delay, 5.0))
            delay *= 2

    if last_error:
        logger.warning(
            "KRW price fetch failed for IAP %s after %d attempts. Last error: %s",
            inapp_id,
            max_attempts,
            last_error,
        )
    else:
        logger.warning(
            "KRW price data unavailable for IAP %s after %d attempts.", inapp_id, max_attempts
        )
    return None


def _get_inapp_purchase_snapshot(
    inapp_id: str,
    *,
    include_relationships: bool = True,
    summary_only: bool = False,
) -> Dict[str, Any]:
    resolved_id, resolved_type = _resolve_inapp_v2_identifier(
        inapp_id, "inAppPurchases"
    )
    lookup_id = resolved_id or inapp_id
    resource_type = (
        resolved_type
        if resolved_id
        else ("inAppPurchasesV2" if lookup_id.isdigit() else "inAppPurchases")
    )

    # Build path candidates
    # Note: resource_type "inAppPurchasesV2" indicates the data model version, not the API endpoint
    # For V2 IAPs (numeric IDs), we use /v2/inAppPurchases/{id}
    # For V1 IAPs (UUIDs), we use /v1/inAppPurchases/{id} or just /inAppPurchases/{id}
    
    path_candidates = []
    if resource_type == "inAppPurchasesV2" or lookup_id.isdigit():
        # This is a V2 IAP (numeric ID) - try V2 API endpoint
        path_candidates = [
            f"/v2/inAppPurchases/{lookup_id}",
            f"/v1/inAppPurchases/{lookup_id}",
            f"/inAppPurchases/{lookup_id}",
        ]
    else:
        # This is a V1 IAP (UUID) - try V1 API endpoints
        path_candidates = [
            f"/v1/inAppPurchases/{lookup_id}",
            f"/inAppPurchases/{lookup_id}",
            f"/v2/inAppPurchases/{lookup_id}",  # Some V1 resources might be accessible via V2
        ]

    last_error: Optional[RuntimeError] = None
    logger.debug("Fetching IAP snapshot for ID=%s, resource_type=%s, trying %d paths", 
                 lookup_id, resource_type, len(path_candidates))
    for idx, path in enumerate(path_candidates, 1):
        logger.debug("Trying path %d/%d: %s for IAP ID=%s", idx, len(path_candidates), path, lookup_id)
        try:
            params: Optional[Dict[str, Any]] = None
            # Note: Apple API has changed, fields and include parameters are no longer reliable
            # Fetch without parameters first, then fetch relationships separately if needed
            if include_relationships:
                # Try with include parameter, but be prepared to fallback
                params = {
                    "include": "inAppPurchaseLocalizations",
                    "fields[inAppPurchaseLocalizations]": "locale,name"
                }
            response = _request("GET", path, params=params)
        except RuntimeError as exc:
            if include_relationships and _is_parameter_error(exc):
                logger.info(
                    "Apple API rejected include when fetching in-app purchase %s via %s; "
                    "falling back to fetch without include.",
                    lookup_id,
                    path,
                )
                # Try the same path without include parameter
                try:
                    base = _request("GET", path)
                    record = _canonicalize_record(base.get("data", {}), {}, {})
                    resource_type = record.get("resourceType", "inAppPurchases")
                    record["localizations"] = _fetch_inapp_localizations(
                        lookup_id, resource_type
                    )
                    # Fetch prices but don't fail if it errors
                    try:
                        record["prices"] = _fetch_inapp_prices(lookup_id, resource_type)
                    except Exception as price_exc:
                        logger.debug("Failed to fetch prices for IAP %s: %s", lookup_id, price_exc)
                        record["prices"] = []
                    return _finalize_inapp_record(
                        record,
                        summary_only=summary_only,
                    )
                except RuntimeError as inner_exc:
                    if (
                        _is_path_error(inner_exc)
                        or _is_forbidden_noop_error(inner_exc)
                        or _is_forbidden_error(inner_exc)
                        or _is_not_found_error(inner_exc)
                    ):
                        last_error = inner_exc
                        logger.debug(
                            "Apple API rejected path %s for in-app purchase %s (inner error: %s); trying alternate path.",
                            path,
                            lookup_id,
                            inner_exc
                        )
                        continue
                    raise
            elif (
                _is_path_error(exc)
                or _is_forbidden_noop_error(exc)
                or _is_forbidden_error(exc)
                or _is_not_found_error(exc)
            ):
                last_error = exc
                logger.debug(
                    "Apple API rejected path %s for in-app purchase %s (error: %s); trying alternate path.",
                    path,
                    lookup_id,
                    exc,
                )
                continue
            else:
                raise
        
        # Successfully fetched with include parameter
        data = response.get("data", {})
        included = (
            response.get("included", []) if include_relationships else []
        )
        localization_map = (
            _index_included(included, "inAppPurchaseLocalizations")
            if include_relationships
            else {}
        )
        prices_map = (
            _index_included(included, "inAppPurchasePrices")
            if include_relationships
            else {}
        )
        result = _canonicalize_record(data, localization_map, prices_map)
        if result:
            if not include_relationships:
                result.pop("localizations", None)
                result.pop("prices", None)
            return _finalize_inapp_record(
                result,
                summary_only=summary_only,
            )
        
        # If canonicalization returned empty, try fetching without include
        if include_relationships:
            try:
                base = _request("GET", path)
                record = _canonicalize_record(base.get("data", {}), {}, {})
                resource_type = record.get("resourceType", "inAppPurchases")
                record["localizations"] = _fetch_inapp_localizations(
                    lookup_id, resource_type
                )
                # Fetch prices but don't fail if it errors
                try:
                    record["prices"] = _fetch_inapp_prices(lookup_id, resource_type)
                except Exception as price_exc:
                    logger.debug("Failed to fetch prices for IAP %s: %s", lookup_id, price_exc)
                    record["prices"] = []
                return _finalize_inapp_record(
                    record,
                    summary_only=summary_only,
                )
            except RuntimeError as inner_exc:
                if (
                    _is_path_error(inner_exc)
                    or _is_forbidden_noop_error(inner_exc)
                    or _is_forbidden_error(inner_exc)
                    or _is_not_found_error(inner_exc)
                ):
                    last_error = inner_exc
                    continue
                raise

    if last_error:
        raise last_error
    raise RuntimeError(
        "Apple API에서 인앱 상품 정보를 가져오지 못했습니다. 다시 시도해 주세요."
    )


def get_inapp_purchase_detail(inapp_id: str) -> Dict[str, Any]:
    return _get_inapp_purchase_snapshot(inapp_id)


def _format_iso8601(date: Optional[str]) -> Optional[str]:
    if not date:
        return None
    try:
        parsed = _dt.datetime.fromisoformat(date.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


APPLE_IAP_LOCALE_OVERRIDES = {
    "ko-kr": "ko",
    "en-sg": "en-GB",
    "zh-tw": "zh-Hant",
}


def _normalize_locale_code(locale: Optional[str]) -> Optional[str]:
    if not locale:
        return None
    trimmed = locale.strip()
    if not trimmed:
        return None
    mapped = APPLE_IAP_LOCALE_OVERRIDES.get(trimmed.lower())
    if mapped:
        return mapped
    return trimmed


def _normalize_localizations(
    localizations: Iterable[Dict[str, Any]],
    base_territory: Optional[str] = None,
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
        locale_code = _normalize_locale_code(data.get("locale"))
        if not locale_code:
            continue
        normalized.append(
            {
                "locale": locale_code,
                "name": data.get("name", ""),
                "description": data.get("description", ""),
                "review_screenshot": review_payload,
            }
        )

    return normalized


def _ensure_localizations(
    inapp_id: str,
    localizations: Iterable[Dict[str, Any]],
    *,
    base_territory: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    normalized = _normalize_localizations(localizations, base_territory)
    resource_type = "inAppPurchasesV2" if inapp_id.isdigit() else "inAppPurchases"
    existing_entries = _list_localization_entries(inapp_id, resource_type)
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
        if inapp_id.isdigit():
            localization_payload = {
                "data": {
                    "type": "inAppPurchaseLocalizations",
                    "attributes": {
                        "name": name,
                        "description": description,
                        "locale": locale,
                    },
                    "relationships": {
                        "inAppPurchaseV2": {"data": {"type": "inAppPurchases", "id": inapp_id}}
                    },
                }
            }
        else:
            localization_payload = {
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
                json=localization_payload,
            )
        else:
            _request("POST", "/inAppPurchaseLocalizations", json=localization_payload)

    refreshed = _list_localization_entries(inapp_id, resource_type)
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


@dataclass
class _PriceTierCacheEntry:
    expires_at: float
    data: List[Dict[str, Any]]


_PRICE_POINT_CACHE: Dict[str, _PricePointCacheEntry] = {}


def _build_price_point_params(
    filters: Dict[str, Any], *, limit: int, cursor: Optional[str] = None
) -> Dict[str, Any]:
    params = dict(filters)
    if _PRICE_POINTS_SUPPORTS_PAGE_PARAMS:
        params["page[limit]"] = limit
        if cursor:
            params["page[cursor]"] = cursor
    else:
        params["limit"] = limit
        if cursor:
            params["cursor"] = cursor
    return params


def _request_price_points(
    filters: Dict[str, Any], *, limit: int, cursor: Optional[str] = None
) -> Dict[str, Any]:
    global _PRICE_POINTS_SUPPORTS_PAGE_PARAMS, _USE_FALLBACK_FOR_PRICE_POINTS

    # Use app-level appPricePoints endpoint instead of inAppPurchasePricePoints
    # This endpoint is more accessible and only returns customer prices (not proceeds)
    app_id = _get_app_id()
    endpoint = f"/apps/{app_id}/appPricePoints"
    
    while True:
        params = _build_price_point_params(filters, limit=limit, cursor=cursor)
        try:
            # Try fallback credentials if previously detected 403 for price points
            if _USE_FALLBACK_FOR_PRICE_POINTS:
                try:
                    return _request("GET", endpoint, params=params, use_fallback=True)
                except AppleStoreConfigError:
                    # Fallback credentials not configured, fall through to primary
                    logger.warning("Fallback credentials not configured, using primary credentials")
                    pass
            
            return _request("GET", endpoint, params=params)
        except AppleStoreApiError as exc:
            if _is_parameter_error(exc) and _PRICE_POINTS_SUPPORTS_PAGE_PARAMS:
                logger.info(
                    "Apple API rejected page[] pagination for price points; "
                    "retrying with compatibility parameters.",
                )
                _PRICE_POINTS_SUPPORTS_PAGE_PARAMS = False
                continue
            
            # Check if it's a 403 forbidden error for price points
            if _is_forbidden_noop_error(exc) and not _USE_FALLBACK_FOR_PRICE_POINTS:
                # Check if fallback credentials are available
                fallback_issuer = _get_fallback_env("APP_STORE_ISSUER_ID")
                fallback_key_id = _get_fallback_env("APP_STORE_KEY_ID")
                fallback_key_path = _get_fallback_env("APP_STORE_PRIVATE_KEY_PATH")
                
                if fallback_issuer and fallback_key_id and fallback_key_path:
                    logger.info(
                        "Price points access forbidden with primary credentials. "
                        "Switching to fallback credentials (App Manager/Finance role)."
                    )
                    _USE_FALLBACK_FOR_PRICE_POINTS = True
                    # Retry with fallback
                    try:
                        return _request("GET", endpoint, params=params, use_fallback=True)
                    except Exception as fallback_exc:
                        logger.error("Fallback credentials also failed: %s", fallback_exc)
                        raise exc  # Raise original exception
            
            raise


def _get_price_point(price_tier: str, territory: str) -> Dict[str, Any]:
    key = _build_price_point_cache_key(price_tier, territory)
    cached = _PRICE_POINT_CACHE.get(key)
    now = time.time()
    if cached and cached.expires_at > now:
        return cached.result

    params = {
        "filter[priceTier]": price_tier,
        "filter[territory]": territory,
    }
    try:
        response = _request_price_points(params, limit=1)
    except AppleStoreApiError as exc:
        if _is_forbidden_error(exc):
            message = _compose_permission_error_message(
                exc, "Apple API 키에 가격 포인트를 조회할 권한이 없습니다."
            )
            raise AppleStorePermissionError(message) from exc
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


def _fetch_all_pages(
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    while True:
        page_params = dict(params or {})
        if cursor:
            page_params["page[cursor]"] = cursor

        response = _request("GET", path, params=page_params or None)
        entries.extend(response.get("data", []) or [])

        cursor = _extract_cursor(response.get("links", {}).get("next"))
        if not cursor:
            break

    return entries


def _price_tier_sort_key(value: str) -> Tuple[int, str]:
    digits = "".join(ch for ch in value if ch.isdigit())
    return (int(digits) if digits else 0, value)


def _get_cached_price_tiers(territory: str) -> Optional[List[Dict[str, Any]]]:
    territory_key = (territory or "KOR").strip().upper()
    now = time.time()

    with _PRICE_TIER_CACHE_LOCK:
        entry = _PRICE_TIER_CACHE.get(territory_key)
        if entry and entry.expires_at > now:
            return [dict(item) for item in entry.data]
        if entry:
            _PRICE_TIER_CACHE.pop(territory_key, None)
    return None


def _set_cached_price_tiers(
    territory: str, tiers: List[Dict[str, Any]], *, ttl: Optional[int] = None
) -> None:
    territory_key = (territory or "KOR").strip().upper()
    expires_at = time.time() + (ttl if ttl is not None else _PRICE_TIER_CACHE_TTL)

    with _PRICE_TIER_CACHE_LOCK:
        _PRICE_TIER_CACHE[territory_key] = _PriceTierCacheEntry(
            expires_at=expires_at, data=[dict(item) for item in tiers]
        )


def _get_price_point_reference_iap_id(*, force_refresh: bool = False) -> Optional[str]:
    global _PRICE_POINT_REFERENCE_IAP_ID

    env_value = os.getenv("APPLE_PRICE_POINT_REFERENCE_IAP_ID", "").strip()
    if env_value:
        return env_value

    with _PRICE_POINT_REFERENCE_LOCK:
        if not force_refresh and _PRICE_POINT_REFERENCE_IAP_ID:
            return _PRICE_POINT_REFERENCE_IAP_ID

    app_id = _get_app_id()
    try:
        response = _request(
            "GET",
            f"/v1/apps/{app_id}/inAppPurchasesV2",
            params={"limit": 1},
        )
    except RuntimeError as exc:
        logger.debug(
            "Failed to fetch reference IAP ID for app %s: %s",
            app_id,
            exc,
        )
        return None

    for entry in response.get("data", []) or []:
        entry_id = entry.get("id")
        if entry_id:
            with _PRICE_POINT_REFERENCE_LOCK:
                _PRICE_POINT_REFERENCE_IAP_ID = entry_id
            return entry_id

    logger.warning(
        "No in-app purchases found for app %s when attempting to cache reference IAP ID",
        app_id,
    )
    return None


def _list_price_points_v2(
    iap_id: str,
    territory: str,
) -> List[Dict[str, Any]]:
    normalized_territory = (territory or "KOR").strip().upper()
    tiers: Dict[str, Dict[str, Any]] = {}
    cursor: Optional[str] = None
    use_page_cursor_param = False

    while True:
        params: Dict[str, Any] = {
            "filter[territory]": normalized_territory,
            "include": "territory",
            "limit": 200,
        }
        if cursor:
            cursor_key = "page[cursor]" if use_page_cursor_param else "cursor"
            params[cursor_key] = cursor

        try:
            response = _request(
                "GET",
                f"/v2/inAppPurchases/{iap_id}/pricePoints",
                params=params,
            )
        except AppleStoreApiError as exc:
            if cursor and _is_parameter_error(exc):
                details = " ".join(
                    [
                        str(entry.get("detail") or entry.get("title") or "").lower()
                        for entry in exc.errors
                    ]
                )
                if "page[cursor]" in details and use_page_cursor_param:
                    logger.info(
                        "Apple API rejected page[cursor] for price points; retrying with cursor parameter."
                    )
                    use_page_cursor_param = False
                    continue
                if " cursor " in f" {details} " and not use_page_cursor_param:
                    logger.info(
                        "Apple API rejected cursor parameter for price points; retrying with page[cursor]."
                    )
                    use_page_cursor_param = True
                    continue
            raise

        included = response.get("included", [])
        territory_map = _index_included(included, "territories")

        for entry in response.get("data", []) or []:
            attributes = entry.get("attributes") or {}
            tier_id = attributes.get("priceTier")
            if not tier_id:
                continue

            currency = attributes.get("currency")
            if not currency:
                relationships = entry.get("relationships", {}) or {}
                territory_data = relationships.get("territory", {}).get("data", {})
                territory_id = (
                    territory_data.get("id") if isinstance(territory_data, dict) else ""
                )
                if territory_id:
                    territory_obj = territory_map.get(territory_id) or {}
                    territory_attrs = (
                        territory_obj.get("attributes", {})
                        if isinstance(territory_obj, dict)
                        else {}
                    )
                    currency = territory_attrs.get("currency")

            tiers[tier_id] = {
                "tier": tier_id,
                "currency": currency,
                "customerPrice": attributes.get("customerPrice"),
                "proceeds": attributes.get("proceeds"),
                "pricePointId": entry.get("id"),
            }

        cursor = _extract_cursor(response.get("links", {}).get("next"))
        if not cursor:
            break

    return [tiers[key] for key in sorted(tiers.keys(), key=_price_tier_sort_key)]


def list_price_tiers(territory: str = "KOR") -> List[Dict[str, Any]]:
    normalized_territory = (territory or "KOR").strip().upper() or "KOR"

    cached = _get_cached_price_tiers(normalized_territory)
    if cached is not None:
        return cached

    last_error: Optional[Exception] = None
    reference_id = _get_price_point_reference_iap_id()

    if reference_id:
        current_id = reference_id
        for attempt in range(2):
            try:
                tiers = _list_price_points_v2(current_id, normalized_territory)
            except AppleStoreApiError as exc:
                last_error = exc
                if _is_not_found_error(exc) and attempt == 0:
                    logger.info(
                        "Reference IAP %s is not valid for price point lookup; refreshing cache.",
                        current_id,
                    )
                    refreshed_id = _get_price_point_reference_iap_id(force_refresh=True)
                    if refreshed_id and refreshed_id != current_id:
                        current_id = refreshed_id
                        continue
                elif _is_forbidden_error(exc):
                    logger.info(
                        "Apple API denied access to price points via /v2 endpoint; "
                        "falling back to app-level price point lookup."
                    )
                else:
                    logger.debug(
                        "Failed to load price tiers via /v2 pricePoints for IAP %s: %s",
                        current_id,
                        exc,
                    )
                break
            except RuntimeError as exc:
                last_error = exc
                logger.debug(
                    "Runtime error while loading price tiers via /v2 pricePoints: %s",
                    exc,
                )
                break
            else:
                _set_cached_price_tiers(normalized_territory, tiers)
                return tiers
    else:
        logger.info(
            "No reference IAP ID available for price point lookup; "
            "falling back to app-level price point APIs."
        )

    tiers = _list_price_tiers_via_app_price_points(normalized_territory)
    if tiers:
        _set_cached_price_tiers(normalized_territory, tiers)
        return tiers

    if not tiers and last_error:
        raise last_error
    return tiers


def _resolve_price_point_for_inapp(
    inapp_id: str,
    territory: str,
    price_krw: Optional[int],
) -> Dict[str, Optional[Any]]:
    """Resolve price point ID from IAP-specific price points (not app-level).
    
    Uses GET /v2/inAppPurchases/{id}/pricePoints?filter[territory]=KOR to get
    all available price points for this specific IAP, then matches the closest one
    to the target KRW price.
    
    Note: We must use IAP-specific price points because app-level price points
    (from /v1/apps/{id}/appPricePoints) have incompatible IDs that cannot be
    used in inAppPurchasePriceSchedules.
    
    Args:
        inapp_id: The IAP ID
        territory: Territory code (e.g., "KOR")
        price_krw: Target price in KRW
        
    Returns:
        Dict with price_point_id, price_tier, matched_price, difference
    """
    normalized_territory = (territory or "KOR").strip().upper() or "KOR"

    if not inapp_id or price_krw is None:
        return {
            "price_point_id": None,
            "price_tier": None,
            "matched_price": None,
            "difference": None,
        }

    try:
        # Use IAP-specific price points (not app-level)
        # This ensures we get compatible price point IDs
        price_points = _list_price_points_v2(inapp_id, normalized_territory)
        logger.info(
            "Fetched %d IAP-specific price points for IAP %s in territory %s (target: %s KRW)",
            len(price_points),
            inapp_id,
            normalized_territory,
            price_krw,
        )
    except AppleStoreApiError as exc:
        logger.warning(
            "Failed to load IAP-specific price points for IAP %s in territory %s: %s",
            inapp_id,
            normalized_territory,
            exc,
        )
        return {
            "price_point_id": None,
            "price_tier": None,
            "matched_price": None,
            "difference": None,
        }
    except Exception as exc:
        logger.error(
            "Unexpected error loading IAP-specific price points for IAP %s in territory %s: %s",
            inapp_id,
            normalized_territory,
            exc,
        )
        return {
            "price_point_id": None,
            "price_tier": None,
            "matched_price": None,
            "difference": None,
        }

    if not price_points:
        logger.warning(
            "No price points available for IAP %s in territory %s. Cannot match price %s KRW.",
            inapp_id,
            normalized_territory,
            price_krw,
        )
        return {
            "price_point_id": None,
            "price_tier": None,
            "matched_price": None,
            "difference": None,
        }

    target_value = Decimal(price_krw)
    best_entry: Optional[Dict[str, Any]] = None
    best_diff: Optional[Decimal] = None

    for entry in price_points:
        price_str = entry.get("customerPrice")
        if not price_str:
            continue
        try:
            value = Decimal(str(price_str))
        except (InvalidOperation, TypeError):
            continue
        diff = abs(value - target_value)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_entry = entry
        elif best_diff is not None and diff == best_diff and best_entry:
            try:
                existing_value = Decimal(str(best_entry.get("customerPrice")))
            except (InvalidOperation, TypeError):
                existing_value = value
            if value < existing_value:
                best_entry = entry

    if not best_entry:
        logger.warning(
            "Unable to match price point for %s KRW in territory %s for IAP %s. "
            "Available price points may not include this exact amount.",
            price_krw,
            normalized_territory,
            inapp_id,
        )
        return {
            "price_point_id": None,
            "price_tier": None,
            "matched_price": None,
            "difference": None,
        }

    try:
        matched_price = int(Decimal(str(best_entry.get("customerPrice"))))
    except (InvalidOperation, TypeError, ValueError):
        matched_price = price_krw

    logger.info(
        "Resolved IAP-specific price point %s (tier %s) for IAP %s in territory %s: %s KRW -> %s KRW (diff: %s)",
        best_entry.get("pricePointId"),
        best_entry.get("tier"),
        inapp_id,
        normalized_territory,
        price_krw,
        matched_price,
        best_diff,
    )

    return {
        "price_point_id": best_entry.get("pricePointId"),
        "price_tier": best_entry.get("tier"),
        "matched_price": matched_price,
        "difference": best_diff,
    }


def _replace_price_schedule(
    inapp_id: str,
    territory: str,
    *,
    price_point_id: Optional[str] = None,
    price_tier: Optional[str] = None,
) -> None:
    def _build_manual_price_entry(
        price_point_identifier: str,
        territory_id: str,
        local_id: str,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Build manual price entry with local ID format required by Apple API.
        
        Apple API requires local IDs in the format '${local-id}' for inline creation.
        """
        territory_value = (territory_id or "KOR").strip().upper() or "KOR"
        relationship = {"type": "inAppPurchasePrices", "id": local_id}
        included = {
            "type": "inAppPurchasePrices",
            "id": local_id,
            "attributes": {
                "startDate": None,
                "endDate": None,
            },
            "relationships": {
                "inAppPurchasePricePoint": {
                    "data": {
                        "type": "inAppPurchasePricePoints",
                        "id": price_point_identifier,
                    }
                },
                "territory": {
                    "data": {
                        "type": "territories",
                        "id": territory_value,
                    }
                },
            },
        }
        return relationship, included

    manual_relationships: List[Dict[str, str]] = []
    included_resources: List[Dict[str, Any]] = []

    if price_point_id:
        # Use local ID format: ${local-id}
        local_id = "${manual-price-1}"
        rel, included = _build_manual_price_entry(price_point_id, territory, local_id)
        manual_relationships.append(rel)
        included_resources.append(included)
    elif price_tier:
        for idx, current_territory in enumerate(_collect_manual_price_territories(territory), start=1):
            try:
                price_point = _get_price_point(price_tier, current_territory)
            except RuntimeError as exc:
                if current_territory == territory.upper():
                    raise
                logger.warning(
                    "Failed to load fixed price point for territory %s: %s",
                    current_territory,
                    exc,
                )
                continue

            point_id = price_point.get("id")
            if not point_id:
                logger.debug(
                    "Price point for territory %s did not include an ID; skipping.",
                    current_territory,
                )
                continue

            # Use local ID format: ${local-id} with index
            local_id = f"${{manual-price-{idx}}}"
            rel, included = _build_manual_price_entry(point_id, current_territory, local_id)
            manual_relationships.append(rel)
            included_resources.append(included)

    if not manual_relationships:
        logger.warning(
            "No manual price entries built for IAP %s; price schedule will not be created.",
            inapp_id,
        )
        return

    logger.info(
        "Creating price schedule for IAP %s with %d manual price entries (base territory: %s)",
        inapp_id,
        len(manual_relationships),
        (territory or "KOR").upper(),
    )

    _clear_existing_manual_prices(inapp_id)

    payload: Dict[str, Any] = {
        "data": {
            "type": "inAppPurchasePriceSchedules",
            "relationships": {
                "inAppPurchase": {
                    "data": {"type": "inAppPurchases", "id": inapp_id}
                },
                "baseTerritory": {
                    "data": {"type": "territories", "id": (territory or "KOR").upper()}
                },
                "manualPrices": {
                    "data": manual_relationships,
                },
            },
        }
    }

    if included_resources:
        payload["included"] = included_resources

    try:
        response = _request("POST", "/v1/inAppPurchasePriceSchedules", json=payload)
    except (AppleStoreApiError, RuntimeError) as exc:
        logger.error(
            "Failed to create price schedule for IAP %s: %s",
            inapp_id,
            exc,
        )
        raise
    else:
        created_id = (
            (response.get("data") or {}).get("id")
            if isinstance(response, dict)
            else None
        )
        if created_id:
            logger.info(
                "Created price schedule %s for IAP %s with %d manual price entries",
                created_id,
                inapp_id,
                len(manual_relationships),
            )


def _normalize_purchase_type_v2(purchase_type: str) -> str:
    if not purchase_type:
        return "CONSUMABLE"
    normalized = purchase_type.replace("-", "_").replace(" ", "_").strip().lower()
    mapping = {
        "consumable": "CONSUMABLE",
        "nonconsumable": "NON_CONSUMABLE",
        "non_consumable": "NON_CONSUMABLE",
        "nonrenewingsubscription": "NON_RENEWING_SUBSCRIPTION",
        "non_renewing_subscription": "NON_RENEWING_SUBSCRIPTION",
        "autorenewablesubscription": "AUTO_RENEWABLE_SUBSCRIPTION",
        "auto_renewable_subscription": "AUTO_RENEWABLE_SUBSCRIPTION",
    }
    return mapping.get(normalized, "CONSUMABLE")


def create_inapp_purchase(
    *,
    product_id: str,
    reference_name: str,
    purchase_type: str,
    cleared_for_sale: bool,
    family_sharable: bool,
    review_note: Optional[str],
    price_point_id: Optional[str],
    price_tier: Optional[str],
    price_krw: Optional[int],
    base_territory: str,
    localizations: Iterable[Dict[str, str]],
) -> Dict[str, Any]:
    purchase_type_v2 = _normalize_purchase_type_v2(purchase_type)
    app_id = _get_app_id()

    v2_payload = {
        "data": {
            "type": "inAppPurchases",
            "attributes": {
                "name": reference_name,
                "productId": product_id,
                "inAppPurchaseType": purchase_type_v2,
                "familySharable": family_sharable,
            },
            "relationships": {
                "app": {
                    "data": {"type": "apps", "id": app_id}
                }
            },
        }
    }

    if review_note is not None:
        v2_payload["data"]["attributes"]["reviewNote"] = review_note

    try:
        result = _request("POST", "/v2/inAppPurchases", json=v2_payload).get("data")
        if not result or not result.get("id"):
            raise RuntimeError("인앱 상품 생성에 실패했습니다.")
    except AppleStoreApiError as exc:
        if not (_is_parameter_error(exc) or _is_path_error(exc)):
            raise
        # Fallback to legacy V1 endpoint
        payload_v1 = {
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
        result = _request("POST", "/inAppPurchases", json=payload_v1).get("data")
        if not result or not result.get("id"):
            raise RuntimeError("인앱 상품 생성에 실패했습니다.")

    inapp_id = result["id"]
    normalized_localizations, locale_ids = _ensure_localizations(
        inapp_id,
        localizations,
        base_territory=base_territory,
    )
    _sync_review_screenshots(locale_ids, normalized_localizations)

    if not price_point_id and price_krw is not None:
        # Resolve price point using the newly created IAP's price points
        resolved = _resolve_price_point_for_inapp(inapp_id, base_territory, price_krw)
        price_point_id = resolved.get("price_point_id") or price_point_id
        price_tier = resolved.get("price_tier") or price_tier

    if price_point_id or price_tier:
        _replace_price_schedule(
            inapp_id,
            base_territory,
            price_point_id=price_point_id,
            price_tier=price_tier,
        )
    availability_territories = _collect_availability_territories(base_territory)
    _update_iap_availability(inapp_id, available_territories=availability_territories)

    return _get_inapp_purchase_snapshot(inapp_id)


def update_inapp_purchase(
    *,
    inapp_id: str,
    reference_name: Optional[str],
    cleared_for_sale: Optional[bool],
    family_sharable: Optional[bool],
    review_note: Optional[str],
    price_point_id: Optional[str],
    price_tier: Optional[str],
    price_krw: Optional[int],
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
        base_territory=base_territory,
    )
    _sync_review_screenshots(locale_ids, normalized_localizations)

    if not price_point_id and price_krw is not None:
        # Resolve price point using the IAP's price points
        resolved = _resolve_price_point_for_inapp(inapp_id, base_territory, price_krw)
        price_point_id = resolved.get("price_point_id") or price_point_id
        price_tier = resolved.get("price_tier") or price_tier

    if price_point_id or price_tier:
        _replace_price_schedule(
            inapp_id,
            base_territory,
            price_point_id=price_point_id,
            price_tier=price_tier,
        )
    availability_territories = _collect_availability_territories(base_territory)
    _update_iap_availability(inapp_id, available_territories=availability_territories)

    return _get_inapp_purchase_snapshot(inapp_id)


def delete_inapp_purchase(inapp_id: str) -> None:
    """Delete an IAP from App Store Connect.
    
    Uses DELETE /v2/inAppPurchases/{id} for V2 IAPs (numeric IDs)
    or DELETE /inAppPurchases/{id} for V1 IAPs (UUIDs).
    """
    # Determine if this is a V2 IAP (numeric ID) or V1 IAP (UUID)
    if inapp_id.isdigit():
        # V2 IAP: use /v2/inAppPurchases/{id}
        _request("DELETE", f"/v2/inAppPurchases/{inapp_id}")
    else:
        # V1 IAP: use /inAppPurchases/{id}
        _request("DELETE", f"/inAppPurchases/{inapp_id}")


def _get_iap_availability_id(inapp_id: str) -> Optional[str]:
    """Get the availability ID for an IAP.
    
    First tries to get it from the IAP's relationships.
    If not found, tries to fetch the availability directly.
    """
    try:
        # Try to get IAP details with relationships
        if inapp_id.isdigit():
            path = f"/v2/inAppPurchases/{inapp_id}"
            params = {"include": "inAppPurchaseAvailability"}
        else:
            path = f"/inAppPurchases/{inapp_id}"
            params = {"include": "inAppPurchaseAvailability"}
        
        response = _request("GET", path, params=params)
        data = response.get("data", {})
        
        # Try to get availability ID from relationship
        relationships = data.get("relationships", {})
        availability_data = relationships.get("inAppPurchaseAvailability", {}).get("data", {})
        availability_id = availability_data.get("id") if isinstance(availability_data, dict) else None
        
        if availability_id:
            logger.debug("Found availability ID %s for IAP %s from relationship", availability_id, inapp_id)
            return availability_id
        
        # If not in relationship, try to find in included resources
        included = response.get("included", [])
        for item in included:
            if item.get("type") == "inAppPurchaseAvailabilities":
                availability_id = item.get("id")
                if availability_id:
                    logger.debug("Found availability ID %s for IAP %s from included resources", availability_id, inapp_id)
                    return availability_id
        
        logger.warning("Could not find availability ID for IAP %s", inapp_id)
        return None
        
    except RuntimeError as exc:
        logger.debug("Failed to get availability ID for IAP %s: %s", inapp_id, exc)
        return None


def _update_iap_availability(inapp_id: str, available_territories: List[str] = None) -> Dict[str, Any]:
    """Update IAP availability by creating a new availability resource.
    
    Note: Apple API does not allow UPDATE operations on inAppPurchaseAvailabilities.
    We must create a new availability resource, which replaces the existing one.
    
    Args:
        inapp_id: The IAP ID (not availability ID)
        available_territories: List of territory codes. If empty list or None, 
                             removes all territories (makes IAP unavailable).
    
    Returns:
        Created availability resource
    """
    # Prepare territories list - empty list means unavailable in all territories
    territories = available_territories if available_territories is not None else []
    
    try:
        # Build payload for POST /v1/inAppPurchaseAvailabilities
        # POST creates a new availability resource (which replaces existing one)
        # DO NOT include id in the payload (it's a CREATE operation)
        # MUST include inAppPurchase relationship
        payload = {
            "data": {
                "type": "inAppPurchaseAvailabilities",
                # No "id" field - this is a CREATE operation
                "attributes": {
                    # Set availableInNewTerritories to False when removing from sale
                    "availableInNewTerritories": False if not territories else True,
                },
                "relationships": {
                    "inAppPurchase": {
                        "data": {
                            "type": "inAppPurchases",
                            "id": inapp_id,
                        }
                    },
                    "availableTerritories": {
                        "data": [
                            {"type": "territories", "id": territory}
                            for territory in territories
                        ]
                    }
                }
            }
        }
        
        # POST to create new availability (replaces existing one)
        response = _request("POST", "/v1/inAppPurchaseAvailabilities", json=payload)
        
        created_data = response.get("data", {})
        created_id = created_data.get("id", "")
        
        logger.info(
            "Created new availability %s for IAP %s: territories=%s (availableInNewTerritories=%s)",
            created_id,
            inapp_id,
            territories if territories else "[] (unavailable)",
            False if not territories else True
        )
        
        return created_data
        
    except RuntimeError as exc:
        logger.error("Failed to create availability for IAP %s: %s", inapp_id, exc)
        raise


def remove_inapp_purchase_from_sale(inapp_id: str) -> Dict[str, Any]:
    """Remove an IAP from sale by creating a new availability with no territories.
    
    This prevents new purchases but maintains access for existing purchasers.
    Creates a new availability resource with empty availableTerritories list,
    which replaces the existing availability.
    """
    try:
        # Create new availability with empty territories list (removes from sale)
        # This replaces the existing availability
        _update_iap_availability(inapp_id, available_territories=[])
        
        # Return updated IAP details
        return _get_inapp_purchase_snapshot(inapp_id, include_relationships=True)
        
    except RuntimeError as exc:
        logger.error("Failed to remove IAP %s from sale: %s", inapp_id, exc)
        # Fallback: try to use clearedForSale if availability update fails
        logger.warning(
            "Availability update failed for IAP %s, falling back to clearedForSale=False",
            inapp_id
        )
        try:
            if inapp_id.isdigit():
                path = f"/v2/inAppPurchases/{inapp_id}"
            else:
                path = f"/inAppPurchases/{inapp_id}"
            
            payload = {
                "data": {
                    "type": "inAppPurchases",
                    "id": inapp_id,
                    "attributes": {
                        "clearedForSale": False,
                    },
                }
            }
            _request("PATCH", path, json=payload)
            return _get_inapp_purchase_snapshot(inapp_id, include_relationships=True)
        except RuntimeError as fallback_exc:
            logger.error("Fallback to clearedForSale also failed for IAP %s: %s", inapp_id, fallback_exc)
            raise exc  # Raise original exception


def _list_price_tiers_via_app_price_points(territory: str = "KOR") -> List[Dict[str, Any]]:
    """List all price points for the app in the given territory.
    
    Uses GET /v1/apps/{id}/appPricePoints?filter[territory]=KOR to fetch
    all available price points directly (not grouped by tier).
    
    Returns a list of price point entries, each with:
    - pricePointId: The price point ID
    - customerPrice: Customer price in the territory currency
    - currency: Currency code
    - tier: Price tier (if available)
    - proceeds: Proceeds (if available)
    """
    def _collect_price_points_from_response(
        price_points: List[Dict[str, Any]], response: Dict[str, Any]
    ) -> None:
        """Collect all price points from API response (no tier grouping)."""
        for entry in response.get("data", []) or []:
            if not isinstance(entry, dict):
                continue

            attributes = entry.get("attributes") or {}
            relationships = entry.get("relationships") or {}
            price_point_id = entry.get("id")
            
            if not price_point_id:
                continue

            currency = attributes.get("currency")
            if not currency:
                territory_data = relationships.get("territory", {}).get("data")
                if isinstance(territory_data, dict):
                    territory_attrs = territory_data.get("attributes", {})
                    if isinstance(territory_attrs, dict):
                        currency = territory_attrs.get("currency")

            # Get tier if available (optional)
            tier_id = attributes.get("priceTier")
            if not tier_id:
                rel_data = relationships.get("priceTier", {}).get("data")
                if isinstance(rel_data, dict):
                    tier_id = rel_data.get("id")

            price_point_entry = {
                "pricePointId": price_point_id,
                "customerPrice": attributes.get("customerPrice"),
                "proceeds": attributes.get("proceeds"),
                "currency": currency,
            }
            
            if tier_id:
                price_point_entry["tier"] = tier_id

            price_points.append(price_point_entry)

    price_points: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    page_count = 0

    permission_message = (
        "Apple API 키에 인앱 가격 정보를 조회할 권한이 없습니다. "
        "일부 API 키는 Admin 역할을 가지고 있더라도 appPricePoints 리소스에 대한 특정 권한이 없을 수 있습니다. "
        "이것은 정상적인 동작이며 애플리케이션은 가격 포인트 데이터 없이도 정상 작동합니다."
    )
    
    global _PRICE_TIER_ACCESS_AVAILABLE
    # If we've previously detected that we don't have permission, return empty list
    # This avoids repeated API calls that will just fail
    if not _PRICE_TIER_ACCESS_AVAILABLE:
        logger.error(
            "Price point access is disabled due to missing permissions "
            "(flag _PRICE_TIER_ACCESS_AVAILABLE=False). "
            "Returning empty price point list. "
            "If you believe you have the correct permissions, restart the server to reset this flag."
        )
        return []
    
    logger.info(
        "Attempting to fetch price points for territory %s via appPricePoints API",
        territory,
    )

    try:
        while True:
            filters = {"filter[territory]": territory}
            response = _request_price_points(filters, limit=200, cursor=cursor)
            page_count += 1
            before_count = len(price_points)
            _collect_price_points_from_response(price_points, response)
            after_count = len(price_points)
            logger.debug(
                "Price point fetch page %d: added %d price points (total: %d)",
                page_count,
                after_count - before_count,
                after_count,
            )
            cursor = _extract_cursor(response.get("links", {}).get("next"))
            if not cursor:
                logger.info(
                    "Successfully fetched %d price points for territory %s in %d pages",
                    len(price_points),
                    territory,
                    page_count,
                )
                break
    except AppleStoreApiError as exc:
        if not _is_forbidden_error(exc):
            raise
        
        # If we get a forbidden error with "no allowed operations", it means
        # the API key doesn't have access to price points at all
        if _is_forbidden_noop_error(exc):
            if _PRICE_TIER_ACCESS_AVAILABLE:
                _PRICE_TIER_ACCESS_AVAILABLE = False
            logger.info(
                "Price point information is not available with current API key permissions. "
                "Even with Admin role, some API keys may not have the specific permission to read appPricePoints. "
                "This is expected behavior and the application will work without price point data."
            )
            return []
        
        message = _compose_permission_error_message(exc, permission_message)
        if _PRICE_TIER_ACCESS_AVAILABLE:
            _PRICE_TIER_ACCESS_AVAILABLE = False
        logger.warning(
            "Apple API permission error for price points: %s. Returning empty list.",
            message,
        )
        return []

    if not price_points:
        logger.warning(
            "Apple API returned 0 price points for territory %s via appPricePoints endpoint.",
            territory,
        )

    # Sort by customerPrice for easier matching
    try:
        price_points.sort(
            key=lambda x: (
                Decimal(str(x.get("customerPrice") or "0")) if x.get("customerPrice") else Decimal("0")
            )
        )
    except (InvalidOperation, TypeError, ValueError):
        # If sorting fails, return as-is
        pass

    return price_points
