import json
import logging
import os
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

PRICE_TEMPLATES_ENV = "PRICE_TEMPLATES"


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


def _parse_template(raw_template: Dict[str, object], *, index: int) -> PriceTemplate:
    if not isinstance(raw_template, dict):
        raise ValueError(f"PRICE_TEMPLATES[{index}] 항목은 객체(JSON object)여야 합니다.")

    template_id = raw_template.get("id")
    if not template_id or not isinstance(template_id, str):
        raise ValueError(f"PRICE_TEMPLATES[{index}]: id 값을 문자열로 지정해야 합니다.")

    label = raw_template.get("label")
    if not label or not isinstance(label, str):
        raise ValueError(f"PRICE_TEMPLATES[{index}]: label 값을 문자열로 지정해야 합니다.")

    default_entry = (
        raw_template.get("default")
        or raw_template.get("default_price")
        or raw_template.get("defaultPrice")
    )
    if not isinstance(default_entry, dict):
        raise ValueError(f"PRICE_TEMPLATES[{index}]: default/defaultPrice 항목이 누락되었거나 객체가 아닙니다.")

    default_price = _normalize_price(default_entry, context=f"PRICE_TEMPLATES[{index}].default")

    regions_entry = raw_template.get("regions") or raw_template.get("prices")
    if regions_entry is None:
        raise ValueError(f"PRICE_TEMPLATES[{index}]: regions/prices 항목이 필요합니다.")
    if not isinstance(regions_entry, dict):
        raise ValueError(f"PRICE_TEMPLATES[{index}]: regions/prices 항목은 객체(JSON object)여야 합니다.")

    region_prices: Dict[str, NormalizedPrice] = {}
    for region_code, price_entry in regions_entry.items():
        if not isinstance(region_code, str):
            raise ValueError(f"PRICE_TEMPLATES[{index}]: 지역 코드는 문자열이어야 합니다.")
        if not isinstance(price_entry, dict):
            raise ValueError(
                f"PRICE_TEMPLATES[{index}]: regions['{region_code}'] 항목은 객체(JSON object)여야 합니다."
            )
        region_prices[region_code] = _normalize_price(
            price_entry, context=f"PRICE_TEMPLATES[{index}].regions['{region_code}']"
        )

    description = raw_template.get("description")
    if description is not None and not isinstance(description, str):
        raise ValueError(f"PRICE_TEMPLATES[{index}]: description 값은 문자열이어야 합니다.")

    return PriceTemplate(
        template_id=template_id,
        label=label,
        default_price=default_price,
        region_prices=region_prices,
        description=description,
    )


@lru_cache(maxsize=1)
def _load_templates() -> tuple[PriceTemplate, ...]:
    raw_value = os.getenv(PRICE_TEMPLATES_ENV)
    if not raw_value:
        logger.warning("가격 템플릿 환경 변수 %s 가 설정되어 있지 않습니다.", PRICE_TEMPLATES_ENV)
        return tuple()

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"PRICE_TEMPLATES 환경 변수 JSON 파싱에 실패했습니다: {exc}") from exc

    if not isinstance(parsed, list):
        raise RuntimeError("PRICE_TEMPLATES 환경 변수는 JSON 배열 형식이어야 합니다.")

    templates: List[PriceTemplate] = []
    template_ids: set[str] = set()
    for index, item in enumerate(parsed):
        template = _parse_template(item, index=index)
        if template.template_id in template_ids:
            raise RuntimeError(f"PRICE_TEMPLATES[{index}]: 중복된 id '{template.template_id}'가 존재합니다.")
        template_ids.add(template.template_id)
        templates.append(template)

    return tuple(templates)


def get_price_templates() -> List[Dict[str, object]]:
    return [template.to_response() for template in _load_templates()]


def get_price_template(template_id: str) -> Optional[PriceTemplate]:
    for template in _load_templates():
        if template.template_id == template_id:
            return template
    return None
