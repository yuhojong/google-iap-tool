# Apple IAP Batch Creation via CSV

## Overview
This feature allows you to batch create multiple Apple In-App Purchases (IAPs) using a CSV file. Unlike Google Play's batch import (which can create/update/delete), this is **create-only** for new IAPs.

## CSV Format

### Required Columns
- `product_id`: The unique product identifier (e.g., com.example.coins100)
- `reference_name`: A human-readable name for internal reference
- `type`: IAP type, must be one of:
  - `consumable`: Can be purchased multiple times
  - `non_consumable`: One-time purchase
  - `non_renewing_subscription`: Time-limited subscription
- `price_tier`: The price tier number (e.g., 1, 5, 10, 20)

### Localization Columns
For each locale you want to support, add two columns:
- `name_{locale}`: Display name in that locale
- `description_{locale}`: Description in that locale

Example locales:
- `en-US`: English (United States)
- `ko`: Korean
- `ja`: Japanese
- `zh-Hans`: Chinese (Simplified)
- `de-DE`: German (Germany)

At least one locale must be provided with both name and description.

## Example CSV

```csv
product_id,reference_name,type,price_tier,name_en-US,description_en-US,name_ko,description_ko
com.example.coins100,100 Coins,consumable,1,100 Coins,Get 100 coins to use in the game,코인 100개,게임에서 사용할 수 있는 코인 100개를 획득하세요
com.example.coins500,500 Coins,consumable,5,500 Coins,Get 500 coins to use in the game,코인 500개,게임에서 사용할 수 있는 코인 500개를 획득하세요
com.example.premium,Premium Features,non_consumable,10,Premium Features,Unlock all premium features,프리미엄 기능,모든 프리미엄 기능을 잠금 해제하세요
```

## Usage

### 1. Preview
POST `/api/apple/inapp/batch/preview`
- Upload CSV file
- Returns validation results and operations to be performed
- Checks for duplicate product IDs
- Validates IAP types

### 2. Apply
POST `/api/apple/inapp/batch/apply`
- Sends operations to create IAPs
- Returns summary of created and failed items
- Failed items include error messages

## Notes

- **Create Only**: This endpoint only creates new IAPs. Existing product IDs will be rejected.
- **Defaults**: All created IAPs will have:
  - `cleared_for_sale`: false (must be enabled manually)
  - `family_sharable`: false
  - `base_territory`: KOR
- **Price Tier**: Make sure the price tier exists in your App Store Connect account
- **Validation**: The preview endpoint validates all data before creation

## Error Handling

Common errors:
- Duplicate product_id in CSV
- Product already exists in App Store Connect
- Invalid IAP type
- Missing required localization
- Invalid price tier

All errors are reported in the response with the row number and product_id for easy troubleshooting.

