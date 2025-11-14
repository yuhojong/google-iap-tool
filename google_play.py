import logging
import re
import os
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

ANDROID_PUBLISHER_SCOPE = "https://www.googleapis.com/auth/androidpublisher"
EEA_WITHDRAWAL_RIGHT_SERVICE = "WITHDRAWAL_RIGHT_SERVICE"


def _build_managed_product_compliance_settings() -> Dict[str, Any]:
    return {"eeaWithdrawalRightType": EEA_WITHDRAWAL_RIGHT_SERVICE}


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


def _get_allowed_regions_override() -> Optional[Set[str]]:
    raw = os.getenv("GOOGLE_PLAY_ALLOWED_REGIONS")
    if not raw:
        return None
    regions: Set[str] = set()
    for token in raw.split(","):
        code = token.strip().upper()
        if len(code) == 2 and code.isalpha():
            regions.add(code)
    if not regions:
        logger.warning("GOOGLE_PLAY_ALLOWED_REGIONS 환경 변수에 유효한 지역 코드가 없습니다.")
        return None
    return regions


def _list_monetization_items(
    resource_name: str, *, page_token: Optional[str] = None, page_size: Optional[int] = None
) -> Dict[str, Any]:
    service = _get_service()
    package_name = _get_package_name()
    params: Dict[str, Any] = {"packageName": package_name}
    if page_token:
        params["pageToken"] = page_token
    if page_size:
        params["pageSize"] = page_size

    try:
        logger.info(
            "Listing monetization resource %s (page_token=%s, page_size=%s)",
            resource_name,
            page_token,
            page_size,
        )
        monetization = service.monetization()
        resource = getattr(monetization, resource_name)()
        response = resource.list(**params).execute()
        items_key = {
            "onetimeproducts": "oneTimeProducts",
            "subscriptions": "subscriptions",
        }.get(resource_name, "items")
        items = response.get(items_key, [])
        logger.info(
            "Fetched %d items from monetization.%s (next_token=%s)",
            len(items or []),
            resource_name,
            response.get("nextPageToken"),
        )
        return response
    except HttpError as exc:
        logger.exception("Google API error while listing monetization resources")
        raise RuntimeError(f"Google API 오류({resource_name}): {exc}") from exc


def _get_monetization_parent() -> str:
    return f"applications/{_get_package_name()}"


def list_onetime_products(page_token: Optional[str] = None) -> Dict[str, Any]:
    response = _list_monetization_items("onetimeproducts", page_token=page_token, page_size=500)
    return {
        "items": response.get("oneTimeProducts", []),
        "nextPageToken": response.get("nextPageToken"),
    }


def iterate_all_onetime_products() -> Iterable[Dict[str, Any]]:
    next_token: Optional[str] = None
    while True:
        result = list_onetime_products(page_token=next_token)
        for item in result.get("items", []):
            yield item
        next_token = result.get("nextPageToken")
        if not next_token:
            break


def get_all_onetime_products() -> List[Dict[str, Any]]:
    return list(iterate_all_onetime_products())


def list_subscription_products(page_token: Optional[str] = None) -> Dict[str, Any]:
    response = _list_monetization_items("subscriptions", page_token=page_token, page_size=500)
    return {
        "items": response.get("subscriptions", []),
        "nextPageToken": response.get("nextPageToken"),
    }


def iterate_all_subscription_products() -> Iterable[Dict[str, Any]]:
    next_token: Optional[str] = None
    while True:
        result = list_subscription_products(page_token=next_token)
        for item in result.get("items", []):
            yield item
        next_token = result.get("nextPageToken")
        if not next_token:
            break


def get_all_subscription_products() -> List[Dict[str, Any]]:
    return list(iterate_all_subscription_products())


def get_all_google_play_products() -> List[Dict[str, Any]]:
    products: List[Dict[str, Any]] = []

    try:
        for onetime in get_all_onetime_products():
            entry = dict(onetime)
            entry["__source"] = "monetization_onetime"
            products.append(entry)
    except Exception as exc:
        logger.warning("Failed to fetch monetization onetime products: %s", exc, exc_info=True)

    try:
        for subscription in get_all_subscription_products():
            entry = dict(subscription)
            entry["__source"] = "monetization_subscription"
            products.append(entry)
    except Exception as exc:
        logger.warning("Failed to fetch monetization subscriptions: %s", exc, exc_info=True)

    logger.info(
        (
            "Aggregated Google Play products: onetime=%d, subscription=%d, total=%d"
        ),
        len([p for p in products if p.get("__source") == "monetization_onetime"]),
        len([p for p in products if p.get("__source") == "monetization_subscription"]),
        len(products),
    )
    return products


def _money_to_price_micros(money: Dict[str, Any]) -> Optional[int]:
    if not isinstance(money, dict):
        return None
    currency = money.get("currencyCode")
    if not currency:
        return None
    units = money.get("units", "0")
    nanos = money.get("nanos", 0)
    try:
        units_int = int(str(units))
        nanos_int = int(str(nanos))
    except ValueError:
        logger.warning("Unexpected money value format: %s", money)
        return None
    return units_int * 1_000_000 + nanos_int // 1_000


def _convert_krw_to_regional_prices(price_won: int) -> Dict[str, Dict[str, str]]:
    if price_won <= 0:
        raise ValueError("가격은 양수여야 합니다.")

    service = _get_service()
    package_name = _get_package_name()
    body = {
        "price": {
            "currencyCode": "KRW",
            "units": str(price_won),
            "nanos": 0,
        }
    }

    try:
        logger.info("Converting KRW price to regional prices via Google API", extra={"price_won": price_won})
        response = (
            service.monetization()
            .convertRegionPrices(packageName=package_name, body=body)
            .execute()
        )
        converted = response.get("convertedRegionPrices") or {}
        prices: Dict[str, Dict[str, str]] = {}
        for region_code, payload in converted.items():
            if not isinstance(region_code, str):
                continue
            price_payload = payload.get("price") if isinstance(payload, dict) else None
            micros = _money_to_price_micros(price_payload or payload)
            currency = None
            if isinstance(price_payload, dict):
                currency = price_payload.get("currencyCode")
            elif isinstance(payload, dict):
                currency = payload.get("currencyCode")
            if micros is None or not currency:
                logger.debug("Skipping region price due to missing data: %s", payload)
                continue
            prices[region_code] = {
                "priceMicros": str(micros),
                "currency": currency,
            }
        logger.info("Generated regional prices for %d regions via convertRegionPrices", len(prices))
        return prices
    except HttpError as exc:
        logger.exception("Google API error during convertRegionPrices")
        raise RuntimeError(f"Google API 오류(convertRegionPrices): {exc}") from exc


def _micros_to_units_nanos(price_micros: int) -> Tuple[str, int]:
    units = price_micros // 1_000_000
    remainder = price_micros % 1_000_000
    nanos = remainder * 1000
    return str(units), nanos


def _extract_non_billable_region(message: str) -> Optional[str]:
    match = re.search(r"Region\s+([A-Z]{2})\s+not\s+billable", message, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def _extract_non_billable_region_from_error(exc: HttpError) -> Optional[str]:
    fragments: List[str] = []
    content = getattr(exc, "content", None)
    if isinstance(content, bytes):
        try:
            fragments.append(content.decode("utf-8", errors="ignore"))
        except Exception:
            pass
    elif isinstance(content, str):
        fragments.append(content)
    if exc.resp and exc.resp.reason:
        fragments.append(str(exc.resp.reason))
    fragments.append(str(exc))
    combined = " ".join(fragment for fragment in fragments if fragment)
    return _extract_non_billable_region(combined)


def _create_legacy_onetime_product(
    *,
    sku: str,
    default_language: str,
    price_won: int,
    price_map: Dict[str, Dict[str, str]],
    translations: Optional[List[Dict[str, str]]],
) -> Dict[str, Any]:
    fallback_price_map = dict(price_map)
    fallback_translations = translations or []

    fallback_default_price: Dict[str, Any]
    if price_won > 0:
        fallback_default_price = {
            "priceMicros": str(price_won * 1_000_000),
            "currency": "KRW",
        }
    else:
        first_region_price = next(iter(fallback_price_map.values()), None)
        if not first_region_price:
            raise RuntimeError("구매 옵션을 구성하기 위한 기본 가격 정보를 찾을 수 없습니다.")
        fallback_default_price = {
            "priceMicros": first_region_price.get("priceMicros", ""),
            "currency": first_region_price.get("currency", ""),
        }

    attempt = 0
    while True:
        attempt += 1
        if not fallback_price_map:
            raise RuntimeError("청구 가능한 지역 가격이 없습니다.")

        fallback_prices = {
            region: {
                "priceMicros": price_info.get("priceMicros", ""),
                "currency": price_info.get("currency", ""),
            }
            for region, price_info in fallback_price_map.items()
            if price_info.get("priceMicros") and price_info.get("currency")
        }
        prices_payload = fallback_prices or None
        try:
            legacy = create_managed_inapp(
                sku=sku,
                default_language=default_language,
                translations=fallback_translations,
                default_price=fallback_default_price,
                prices=prices_payload,
                status="active",
            )
            legacy["__source"] = "legacy_inappproduct"
            return legacy
        except (HttpError, RuntimeError) as exc:
            # HttpError or RuntimeError wrapping HttpError
            http_error: Optional[HttpError] = None
            if isinstance(exc, HttpError):
                http_error = exc
            elif isinstance(exc, RuntimeError) and exc.__cause__ and isinstance(exc.__cause__, HttpError):
                http_error = exc.__cause__
            
            if http_error:
                region_code = _extract_non_billable_region_from_error(http_error)
                if region_code and region_code in fallback_price_map:
                    logger.info(
                        "Removing non-billable region '%s' and retrying legacy creation (attempt %d)",
                        region_code,
                        attempt,
                    )
                    fallback_price_map.pop(region_code, None)
                    continue
            raise


def create_onetime_product(
    *,
    sku: str,
    default_language: str,
    price_won: int,
    regional_pricing: Optional[Dict[str, Any]] = None,
    translations: Optional[List[Dict[str, str]]] = None,
    allowed_regions: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    if price_won <= 0 and not regional_pricing:
        raise ValueError("가격은 양수여야 합니다.")

    price_map: Dict[str, Dict[str, str]]
    if regional_pricing and isinstance(regional_pricing, dict):
        base_prices = regional_pricing.get("prices")
        if isinstance(base_prices, dict) and base_prices:
            price_map = {}
            for region_code, price_payload in base_prices.items():
                if not isinstance(region_code, str) or not isinstance(price_payload, dict):
                    continue
                price_micros = price_payload.get("priceMicros")
                currency = price_payload.get("currency")
                if not price_micros or not currency:
                    continue
                price_map[region_code] = {
                    "priceMicros": str(price_micros),
                    "currency": currency,
                }
        else:
            price_map = _convert_krw_to_regional_prices(price_won)
    else:
        price_map = _convert_krw_to_regional_prices(price_won)
    if not price_map:
        raise RuntimeError("convertRegionPrices 결과가 비어 있습니다.")

    allowed_override = _get_allowed_regions_override()
    allowed_set: Optional[Set[str]] = None
    if allowed_override:
        allowed_set = allowed_override
    elif allowed_regions:
        normalized = {
            str(code).strip().upper() for code in allowed_regions if isinstance(code, str)
        }
        allowed_set = {code for code in normalized if len(code) == 2 and code.isalpha()}
        if normalized and not allowed_set:
            logger.warning("허용된 지역 목록에 유효한 ISO 코드가 없어 필터링을 건너뜁니다.")
    if allowed_set:
        filtered_map = {
            region: info
            for region, info in price_map.items()
            if isinstance(region, str) and region.strip().upper() in allowed_set
        }
        removed = len(price_map) - len(filtered_map)
        if filtered_map:
            if removed:
                logger.info(
                    "허용된 지역 목록을 적용하여 %d개 지역 가격을 제외했습니다 (SKU: %s).",
                    removed,
                    sku,
                )
            price_map = filtered_map
        else:
            logger.warning(
                "허용된 지역 목록을 적용하면 모든 지역 가격이 제거되어 원본 지역 목록을 사용합니다 (SKU: %s).",
                sku,
            )

    regional_configs: List[Dict[str, Any]] = []
    for region_code, price_info in price_map.items():
        try:
            price_micros = int(price_info.get("priceMicros", "0"))
        except (TypeError, ValueError):
            logger.debug("Invalid priceMicros for region %s: %s", region_code, price_info)
            continue
        currency = price_info.get("currency")
        if not currency or price_micros <= 0:
            continue
        units, nanos = _micros_to_units_nanos(price_micros)
        regional_configs.append(
            {
                "regionCode": region_code,
                "availability": "AVAILABLE",
                "price": {
                    "currencyCode": currency,
                    "units": units,
                    "nanos": nanos,
                },
            }
        )

    if not regional_configs:
        raise RuntimeError("구매 옵션을 구성하기 위한 지역 가격 정보를 찾을 수 없습니다.")

    listings_payload: List[Dict[str, str]] = []
    default_listing: Optional[Dict[str, str]] = None
    for item in translations or []:
        language = item.get("language")
        title = item.get("title")
        description = item.get("description")
        if not language or not title or not description:
            raise ValueError("모든 번역 항목에는 언어, 이름, 설명이 필요합니다.")
        listing_entry = {
            "languageCode": language,
            "title": title,
            "description": description,
        }
        listings_payload.append(listing_entry)
        if language == default_language:
            default_listing = listing_entry

    if not default_listing and listings_payload:
        raise ValueError("기본 언어 번역 정보를 찾을 수 없습니다.")

    service = _get_service()
    package_name = _get_package_name()

    request_body: Dict[str, Any] = {
        "requests": [
            {
                "allowMissing": True,
                "updateMask": "listings,purchaseOptions",
                "oneTimeProduct": {
                    "name": f"applications/{package_name}/oneTimeProducts/{sku}",
                    "listings": listings_payload,
                    "purchaseOptions": [
                        {
                            "purchaseOptionId": "default",
                            "state": "ACTIVE",
                            "regionalPricingAndAvailabilityConfigs": regional_configs,
                        }
                    ],
                },
            }
        ]
    }

    try:
        response = (
            service.monetization()
            .onetimeproducts()
            .batchUpdate(packageName=package_name, body=request_body)
            .execute()
        )
        updated_products: List[Dict[str, Any]] = []
        if isinstance(response, dict):
            updated_products = response.get("oneTimeProducts") or []
            if not updated_products:
                responses = response.get("responses") or []
                for res in responses:
                    product = res.get("oneTimeProduct") or res.get("one_time_product")
                    if isinstance(product, dict):
                        updated_products.append(product)
        if not updated_products:
            raise RuntimeError("Google API 응답에 생성된 상품 정보가 없습니다.")
        created_product = dict(updated_products[0])
        created_product["__source"] = "monetization_onetime"
        created_product.setdefault("sku", sku)
        created_product.setdefault("oneTimeProductId", sku)
        return created_product
    except HttpError as exc:
        if exc.resp.status == 400:
            logger.warning(
                "Monetization onetimeproducts.batchUpdate failed for '%s' (%s). Falling back to legacy creation.",
                sku,
                exc,
            )
            return _create_legacy_onetime_product(
                sku=sku,
                default_language=default_language,
                price_won=price_won,
                price_map=price_map,
                translations=translations,
            )
        logger.exception("Google API error while creating monetization onetime product")
        raise RuntimeError(f"Google API 오류(onetimeproducts.batchUpdate): {exc}") from exc


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
        
        # Auto-populate regional prices using Google conversion API
        try:
            resolved_prices = _convert_krw_to_regional_prices(price_won)
            if not resolved_prices:
                logger.warning(
                    "convertRegionPrices returned no regional prices; creation may fail if app is available in other regions"
                )
        except Exception as exc:
            logger.error("Failed to auto-populate regional prices via convertRegionPrices", exc_info=True)
            raise

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
        "managedProductTaxesAndComplianceSettings": _build_managed_product_compliance_settings(),
    }
    if resolved_prices:
        body["prices"] = resolved_prices
    
    # Debug: Log what we're sending to Google API
    logger.info(f"Creating IAP '{sku}' with {len(resolved_prices) if resolved_prices else 0} regional prices")
    if not resolved_prices:
        logger.warning(f"⚠️ No regional prices set for '{sku}' - this will likely fail!")
    
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
        "managedProductTaxesAndComplianceSettings": _build_managed_product_compliance_settings(),
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
