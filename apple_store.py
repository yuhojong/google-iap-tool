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
import signal
import sys
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
_APPLE_API_TIMEOUT = int(os.getenv("APPLE_API_TIMEOUT", "60"))  # Increased default timeout

_TOKEN_LOCK = threading.Lock()
_TOKEN_CACHE: Optional[Tuple[str, int]] = None

# Fallback token cache for price points API (with App Manager/Finance permissions)
_FALLBACK_TOKEN_CACHE: Optional[Tuple[str, int]] = None
_USE_FALLBACK_FOR_PRICE_POINTS = False

# Global flag for graceful shutdown
_FETCH_INTERRUPTED = threading.Event()

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
# appPricePoints endpoint uses simple limit/cursor params, not page[limit]
_PRICE_POINTS_SUPPORTS_PAGE_PARAMS = False

_INAPP_V2_ID_CACHE: Dict[str, Optional[str]] = {}
_INAPP_V2_ID_BY_PRODUCT_ID: Dict[str, Optional[str]] = {}

_LOCALIZATION_LIST_STRATEGY: Optional[str] = None


_PRICE_TIER_GUESS_RANGE = tuple(str(value) for value in range(0, 201))


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
    """Setup interrupt handler for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info("Interrupt signal received. Shutting down gracefully...")
        _FETCH_INTERRUPTED.set()
        # Give time for current operation to complete, then exit
        def delayed_exit():
            time.sleep(2)
            logger.info("Exiting after graceful shutdown")
            sys.exit(0)
        
        exit_thread = threading.Thread(target=delayed_exit, daemon=True)
        exit_thread.start()
    
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)


def check_interrupted() -> bool:
    """Check if interrupt signal has been received."""
    return _FETCH_INTERRUPTED.is_set()


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
            
            def fetch_iap_with_retry(identifier: Tuple[str, str], idx: int, max_retries: int = 5) -> Optional[Dict[str, Any]]:
                inapp_id, resource_type = identifier
                for attempt in range(max_retries):
                    try:
                        record = _get_inapp_purchase_snapshot(
                            inapp_id, include_relationships=include_relationships
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
                )
                return items, next_cursor, total_count
            
            # Return partial results if we had some successes
            return items, next_cursor, total_count

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
                if not include_relationships:
                    item.pop("localizations", None)
                items.append(item)

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
        item["prices"] = _fetch_inapp_prices(inapp_id, resource_type)
        items.append(item)

    next_cursor = _extract_cursor(response.get("links", {}).get("next"))
    return items, next_cursor, None  # No total count available in basic mode


def iterate_all_inapp_purchases(
    limit: int = 200, *, include_relationships: bool = True
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
        )
        
        cumulative_count += len(items)
        
        # Yield items
        for item in items:
            yield item
        
        # Stop if no more cursor OR if we received 0 items (completed)
        if not cursor or len(items) == 0:
            if cumulative_count > 0:
                logger.info("Fetch complete: %d total items retrieved", cumulative_count)
            break


def get_all_inapp_purchases(
    *, include_relationships: bool = True
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    items = list(
        iterate_all_inapp_purchases(include_relationships=include_relationships)
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
            
            # Fetch all pages of IDs only (no details)
            all_ids = [iap_id for iap_id, _ in identifiers]
            
            while next_cursor:
                identifiers, next_cursor, _ = _list_inapp_purchase_identifiers_v2(
                    cursor=next_cursor, limit=200
                )
                all_ids.extend([iap_id for iap_id, _ in identifiers])
            
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

    # For V2 IAPs (numeric IDs), use /v2/inAppPurchases/{id}/pricePoints with include=priceTier,territory
    # For V1 IAPs (UUIDs), use /inAppPurchases/{id}/prices
    paths = []
    if resource_type == "inAppPurchasesV2":
        paths.append((f"/v2/inAppPurchases/{inapp_id}/pricePoints", True))  # (path, is_v2)
    paths.append((f"/inAppPurchases/{inapp_id}/prices", False))

    for path, is_v2 in paths:
        try:
            # V2 endpoint needs include parameter to get priceTier and territory
            params = {"include": "priceTier,territory"} if is_v2 else None
            response = _request("GET", path, params=params)
        except RuntimeError as exc:
            if (
                _is_path_error(exc)
                or _is_forbidden_noop_error(exc)
                or _is_forbidden_error(exc)
                or _is_not_found_error(exc)
            ):
                continue
            raise
        else:
            prices: List[Dict[str, Any]] = []
            if is_v2:
                # V2 response includes related priceTier and territory in "included"
                included = response.get("included", [])
                price_tier_map = _index_included(included, "inAppPurchasePriceTiers")
                territory_map = _index_included(included, "territories")
                
                for entry in response.get("data", []) or []:
                    parsed = _parse_price_point_entry(entry, price_tier_map, territory_map)
                    if parsed:
                        prices.append(parsed)
            else:
                # V1 response has prices directly
                for entry in response.get("data", []) or []:
                    parsed = _parse_price_entry(entry)
                    if parsed:
                        prices.append(parsed)
            return prices

    if _INAPP_PRICE_RELATIONSHIP_AVAILABLE:
        logger.warning(
            "Apple API rejected in-app purchase price relationship lookup; "
            "price data will not be included in responses."
        )
    _INAPP_PRICE_RELATIONSHIP_AVAILABLE = False
    return []


def _get_inapp_purchase_snapshot(
    inapp_id: str, *, include_relationships: bool = True
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
            # V2 API uses different include parameter
            if path.startswith("/v2/"):
                params = (
                    {
                        "include": "inAppPurchaseLocalizations"
                    }
                    if include_relationships
                    else None
                )
            else:
                params = (
                    {
                        "include": "inAppPurchaseLocalizations,inAppPurchasePrices"
                    }
                    if include_relationships
                    else None
                )
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
                    record["prices"] = _fetch_inapp_prices(lookup_id, resource_type)
                    return record
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
            return result
        
        # If canonicalization returned empty, try fetching without include
        if include_relationships:
            try:
                base = _request("GET", path)
                record = _canonicalize_record(base.get("data", {}), {}, {})
                resource_type = record.get("resourceType", "inAppPurchases")
                record["localizations"] = _fetch_inapp_localizations(
                    lookup_id, resource_type
                )
                record["prices"] = _fetch_inapp_prices(lookup_id, resource_type)
                return record
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

    territories: List[str] = [territory.upper()]
    for fixed_territory in _FIXED_PRICE_TERRITORIES:
        normalized = fixed_territory.upper()
        if normalized and normalized not in territories:
            territories.append(normalized)

    for current_territory in territories:
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
            # Note: appPricePoints endpoint only returns customerPrice (not proceeds)
            tiers[tier_id] = {
                "tier": tier_id,
                "currency": attributes.get("currency"),
                "customerPrice": attributes.get("customerPrice"),
            }

    tiers: Dict[str, Dict[str, Any]] = {}
    cursor: Optional[str] = None

    permission_message = (
        "Apple API 키에 인앱 가격 정보를 조회할 권한이 없습니다. "
        "일부 API 키는 Admin 역할을 가지고 있더라도 inAppPurchasePricePoints 리소스에 대한 특정 권한이 없을 수 있습니다. "
        "이것은 정상적인 동작이며 애플리케이션은 가격 티어 데이터 없이도 정상 작동합니다."
    )
    
    # If we've previously detected that we don't have permission, return empty list
    # This avoids repeated API calls that will just fail
    if not _INAPP_PRICE_RELATIONSHIP_AVAILABLE:
        logger.warning(
            "Price tier access is disabled due to missing permissions. "
            "Returning empty price tier list."
        )
        return []

    try:
        while True:
            filters = {"filter[territory]": territory}
            response = _request_price_points(filters, limit=200, cursor=cursor)
            _collect_from_response(tiers, response)
            cursor = _extract_cursor(response.get("links", {}).get("next"))
            if not cursor:
                break
    except AppleStoreApiError as exc:
        if not _is_forbidden_error(exc):
            raise
        
        # If we get a forbidden error with "no allowed operations", it means
        # the API key doesn't have access to price points at all
        if _is_forbidden_noop_error(exc):
            logger.info(
                "Price tier information is not available with current API key permissions. "
                "Even with Admin role, some API keys may not have the specific permission to read inAppPurchasePricePoints. "
                "This is expected behavior and the application will work without price tier data."
            )
            return []
        
        message = _compose_permission_error_message(exc, permission_message)
        logger.warning(
            "Apple API permission error for price points; attempting tier enumeration as fallback."
        )

        tiers.clear()
        chunk_size = 25
        for index in range(0, len(_PRICE_TIER_GUESS_RANGE), chunk_size):
            chunk = _PRICE_TIER_GUESS_RANGE[index : index + chunk_size]
            try:
                filters = {
                    "filter[territory]": territory,
                    "filter[priceTier]": ",".join(chunk),
                }
                response = _request_price_points(filters, limit=200)
            except AppleStoreApiError as inner_exc:
                if _is_parameter_error(inner_exc):
                    logger.debug(
                        "Ignoring parameter error when probing price tiers chunk %s",
                        chunk,
                    )
                    continue
                if _is_forbidden_noop_error(inner_exc):
                    # No permissions for price points at all
                    logger.warning(
                        "No permission to access price points; returning empty tier list."
                    )
                    return []
                if _is_forbidden_error(inner_exc):
                    logger.debug(
                        "Apple API forbade chunked price tier request; retrying tier-by-tier",
                        extra={"chunk": chunk},
                    )
                    successful = False
                    for tier in chunk:
                        try:
                            tier_filters = {
                                "filter[territory]": territory,
                                "filter[priceTier]": tier,
                            }
                            tier_response = _request_price_points(
                                tier_filters, limit=1
                            )
                        except AppleStoreApiError as single_exc:
                            if _is_parameter_error(single_exc):
                                logger.debug(
                                    "Ignoring parameter error when probing price tier %s",
                                    tier,
                                )
                                continue
                            if _is_forbidden_noop_error(single_exc):
                                # If no permissions for price points, return empty list
                                logger.warning(
                                    "No permission to access price points; returning empty tier list."
                                )
                                return []
                            if _is_forbidden_error(single_exc):
                                message = _compose_permission_error_message(
                                    single_exc, permission_message
                                )
                                raise AppleStorePermissionError(message) from single_exc
                            raise
                        _collect_from_response(tiers, tier_response)
                        successful = True
                    if successful:
                        continue
                    message = _compose_permission_error_message(
                        inner_exc, permission_message
                    )
                    raise AppleStorePermissionError(message) from inner_exc
                raise
            _collect_from_response(tiers, response)

    return [tiers[key] for key in sorted(tiers.keys(), key=_tier_sort_key)]
