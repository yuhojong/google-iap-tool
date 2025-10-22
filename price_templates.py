import logging
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NormalizedPrice:
    currency: str
    price: str
    price_micros: str

    def as_api_dict(self) -> Dict[str, str]:
        return {"currency": self.currency, "priceMicros": self.price_micros}

    def as_response_dict(self) -> Dict[str, str]:
        return {
            "currency": self.currency,
            "price": self.price,
            "priceMicros": self.price_micros,
        }


@dataclass(frozen=True)
class PriceTemplate:
    template_id: str
    label: str
    default_price: NormalizedPrice
    region_prices: Dict[str, NormalizedPrice]
    description: Optional[str] = None

    def to_response(self) -> Dict[str, object]:
        regions: List[Dict[str, str]] = []
        for region_code, price in self.region_prices.items():
            region_entry = {"region": region_code}
            region_entry.update(price.as_response_dict())
            regions.append(region_entry)
        regions.sort(key=lambda entry: entry["region"])
        response: Dict[str, object] = {
            "id": self.template_id,
            "label": self.label,
            "default_currency": self.default_price.currency,
            "default_price": self.default_price.price,
            "default_price_micros": self.default_price.price_micros,
            "regions": regions,
        }
        if self.description:
            response["description"] = self.description
        return response

    def to_pricing_payload(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "defaultPrice": self.default_price.as_api_dict(),
        }
        if self.region_prices:
            payload["prices"] = {
                region: price.as_api_dict() for region, price in self.region_prices.items()
            }
        return payload


def _normalize_price(entry: Dict[str, object], *, context: str) -> NormalizedPrice:
    currency = entry.get("currency")
    if not currency or not isinstance(currency, str):
        raise ValueError(f"{context}: currency 값을 문자열로 지정해야 합니다.")
    currency = currency.upper()

    price_value = entry.get("price")
    price_micros_value = entry.get("priceMicros") or entry.get("micros")

    if price_value is None and price_micros_value is None:
        raise ValueError(f"{context}: price 또는 priceMicros 중 하나는 반드시 지정해야 합니다.")

    if price_value is not None:
        try:
            decimal_value = Decimal(str(price_value))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"{context}: price 값을 숫자로 변환할 수 없습니다.") from exc
        if decimal_value <= 0:
            raise ValueError(f"{context}: price 값은 0보다 커야 합니다.")
        price_micros = (decimal_value * Decimal("1000000")).quantize(Decimal("1"))
        price_str = format(decimal_value.normalize(), 'f')
        price_micros_str = str(int(price_micros))
    else:
        try:
            price_micros_int = int(price_micros_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}: priceMicros 값은 정수여야 합니다.") from exc
        if price_micros_int <= 0:
            raise ValueError(f"{context}: priceMicros 값은 0보다 커야 합니다.")
        price_micros_str = str(price_micros_int)
        price_decimal = (Decimal(price_micros_str) / Decimal("1000000")).quantize(Decimal("0.000001")).normalize()
        price_str = format(price_decimal, 'f')

    return NormalizedPrice(currency=currency, price=price_str, price_micros=price_micros_str)


def _format_decimal_for_display(value: Decimal) -> str:
    text = f"{value:,.6f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def generate_price_templates_from_products(
    products: Iterable[Dict[str, object]]
) -> List[PriceTemplate]:
    grouped: Dict[str, Dict[str, object]] = {}

    for raw_product in products:
        if not isinstance(raw_product, dict):
            continue

        sku = raw_product.get("sku") or ""
        default_entry = raw_product.get("defaultPrice")
        if not isinstance(default_entry, dict):
            logger.debug("상품 '%s'의 기본 가격 정보를 확인할 수 없어 템플릿 생성을 건너뜁니다.", sku)
            continue

        try:
            default_price = _normalize_price(default_entry, context=f"상품 '{sku}' 기본 가격")
        except ValueError as exc:
            logger.warning("상품 '%s' 기본 가격을 정규화하는 데 실패했습니다: %s", sku, exc)
            continue

        if default_price.currency.upper() != "KRW":
            logger.debug("상품 '%s' 기본 통화가 KRW가 아니므로 템플릿에서 제외합니다.", sku)
            continue

        key = default_price.price_micros
        record = grouped.setdefault(
            key,
            {
                "default": default_price,
                "regions": None,
                "valid": True,
                "skus": [],
            },
        )
        record["skus"].append(sku)

        prices_entry = raw_product.get("prices")
        region_prices: Dict[str, NormalizedPrice] = {}
        if prices_entry is not None:
            if not isinstance(prices_entry, dict):
                logger.warning("상품 '%s'의 지역 가격 정보가 올바르지 않아 템플릿에서 제외합니다.", sku)
                record["valid"] = False
                continue
            region_valid = True
            for region_code, price_entry in prices_entry.items():
                if not isinstance(region_code, str) or not isinstance(price_entry, dict):
                    region_valid = False
                    break
                try:
                    normalized_region = _normalize_price(
                        price_entry,
                        context=f"상품 '{sku}' 지역 '{region_code}' 가격",
                    )
                except ValueError as exc:
                    logger.warning(
                        "상품 '%s'의 지역 '%s' 가격을 정규화하는 데 실패했습니다: %s",
                        sku,
                        region_code,
                        exc,
                    )
                    region_valid = False
                    break
                region_prices[region_code] = normalized_region

            if not region_valid:
                record["valid"] = False
                continue

        if record["regions"] is None:
            record["regions"] = region_prices
        elif record["regions"] != region_prices:
            logger.info(
                "KRW price micros %s 그룹의 지역 가격 정보가 일치하지 않아 템플릿에서 제외합니다.",
                key,
            )
            record["valid"] = False

    templates: List[PriceTemplate] = []
    for key, record in grouped.items():
        if not record.get("valid"):
            continue

        default_price = record["default"]
        region_prices = record.get("regions") or {}
        try:
            decimal_price = (Decimal(default_price.price_micros) / Decimal("1000000")).normalize()
        except (InvalidOperation, ValueError) as exc:
            logger.warning(
                "KRW price micros %s 값을 변환하는 데 실패하여 템플릿 생성을 건너뜁니다: %s",
                key,
                exc,
            )
            continue
        label = f"₩{_format_decimal_for_display(decimal_price)}"
        skus: List[str] = [sku for sku in record.get("skus", []) if sku]
        description: Optional[str] = None
        if skus:
            description = f"{len(skus)}개 상품에서 추출된 가격 템플릿"

        templates.append(
            PriceTemplate(
                template_id=f"krw-{key}",
                label=label,
                default_price=default_price,
                region_prices=region_prices,
                description=description,
            )
        )

    templates.sort(key=lambda tpl: int(tpl.default_price.price_micros))
    return templates


def get_template_by_id(
    templates: Iterable[PriceTemplate], template_id: str
) -> Optional[PriceTemplate]:
    for template in templates:
        if template.template_id == template_id:
            return template
    return None


def index_templates_by_price_micros(
    templates: Iterable[PriceTemplate],
) -> Dict[str, PriceTemplate]:
    mapping: Dict[str, PriceTemplate] = {}
    for template in templates:
        if template.default_price.currency.upper() != "KRW":
            continue
        mapping[template.default_price.price_micros] = template
    return mapping
