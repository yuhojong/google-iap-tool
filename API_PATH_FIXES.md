# Apple App Store API Path Fixes

## Issues Fixed

### 1. **Price Points 403 Forbidden Error**
**Problem:** The `inAppPurchasePricePoints` endpoint returned 403 errors even with Admin role API keys, and fallback credentials also failed.

**Root Cause:** The `inAppPurchasePricePoints` endpoint requires very specific permissions that not all API keys have, even with Admin role.

**Solution:** Switched to the app-level `appPricePoints` endpoint:
- **Old (BROKEN):** `GET /v1/inAppPurchasePricePoints?filter[territory]=...`
- **New (WORKING):** `GET /v1/apps/{APP_ID}/appPricePoints?filter[territory]=...`

**Benefits:**
- ✅ More accessible endpoint (works with standard API keys)
- ✅ Returns `customerPrice` (user-facing prices)
- ⚠️ Does NOT return `proceeds` (developer revenue) - removed from response

---

### 2. **V2 IAP Localization 404 Errors**
**Problem:** When editing V2 IAPs (numeric IDs), localization fetch failed with 404 errors:
```
[ERROR] Apple API error 404: [NOT_FOUND] The path provided does not match a defined resource type. 
| URL: https://api.appstoreconnect.apple.com/v1/inAppPurchasesV2/6737119376/inAppPurchaseLocalizations
```

**Root Cause:** Incorrect API paths for V2 IAPs. The code was using `/v1/inAppPurchasesV2/...` which doesn't exist.

**Solution:** Fixed API paths for V2 IAPs:
- **Old (BROKEN):** `/v1/inAppPurchasesV2/{id}/inAppPurchaseLocalizations`
- **New (WORKING):** `/v2/inAppPurchases/{id}/inAppPurchaseLocalizations`

---

### 3. **V2 IAP Price 404 Errors**
**Problem:** When editing V2 IAPs, price fetch failed with 404 errors:
```
[ERROR] Apple API error 404: [NOT_FOUND] The path provided does not match a defined resource type. 
| URL: https://api.appstoreconnect.apple.com/v1/inAppPurchasesV2/6737119376/inAppPurchasePrices
```

**Root Cause:** Incorrect API paths for V2 IAP prices.

**Solution:** Fixed API paths for V2 IAP prices with proper include parameter:
- **Old (BROKEN):** `/v1/inAppPurchasesV2/{id}/inAppPurchasePrices`
- **New (WORKING):** `/v2/inAppPurchases/{id}/pricePoints?include=priceTier,territory`

The V2 endpoint returns pricePoints with relationships that need to be included for easy mapping.

---

### 4. **V2 IAP Include Parameter Error**
**Problem:** When fetching V2 IAP details, got parameter errors:
```
[ERROR] Apple API error 400: [PARAMETER_ERROR.INVALID] 'inAppPurchasePrices' is not a valid relationship name 
| URL: https://api.appstoreconnect.apple.com/v2/inAppPurchases/6737119376
```

**Root Cause:** The `include` parameter value was wrong for V2 endpoint.

**Solution:** The code already had fallback logic to retry without the `include` parameter, which works correctly.

---

## API Path Reference

### **Correct V2 IAP Paths** (Numeric IDs like `6737119376`)

| Resource | Correct Path | Wrong Path (404) |
|----------|--------------|------------------|
| **IAP Detail** | `/v2/inAppPurchases/{id}?include=inAppPurchaseLocalizations` | `/v1/inAppPurchasesV2/{id}` ❌ |
| **Localizations** | `/v2/inAppPurchases/{id}/inAppPurchaseLocalizations` | `/v1/inAppPurchasesV2/{id}/inAppPurchaseLocalizations` ❌ |
| **Prices** | `/v2/inAppPurchases/{id}/pricePoints?include=priceTier,territory` | `/v1/inAppPurchasesV2/{id}/inAppPurchasePrices` ❌ |

### **Correct V1 IAP Paths** (UUIDs like `a1b2c3d4-...`)

| Resource | Correct Path |
|----------|--------------|
| **IAP Detail** | `/v1/inAppPurchases/{id}` or `/inAppPurchases/{id}` |
| **Localizations** | `/inAppPurchases/{id}/inAppPurchaseLocalizations` |
| **Prices** | `/inAppPurchases/{id}/prices` |

### **Price Tier Endpoints**

| Endpoint | Access Level | Returns | Pagination |
|----------|--------------|---------|------------|
| `/v1/inAppPurchasePricePoints` ❌ | Requires special permissions | `customerPrice`, `proceeds` | `page[limit]`, `page[cursor]` |
| `/v1/apps/{APP_ID}/appPricePoints` ✅ | Standard API key | `customerPrice` only | `limit`, `cursor` |

**Note:** The `appPricePoints` endpoint uses simple `limit`/`cursor` parameters, NOT `page[limit]`/`page[cursor]`.

---

## Code Changes

### **1. Price Points Endpoint** (`_request_price_points`)

```python
# Before (403 Forbidden):
endpoint = "/inAppPurchasePricePoints"
params = {"page[limit]": 200, "filter[territory]": "KOR"}

# After (Works):
app_id = _get_app_id()
endpoint = f"/apps/{app_id}/appPricePoints"
params = {"limit": 200, "filter[territory]": "KOR"}  # Simple limit, not page[limit]
```

**Pagination Parameter Change:**
- Old endpoint: `page[limit]`, `page[cursor]`
- New endpoint: `limit`, `cursor` (simple params)
- Set `_PRICE_POINTS_SUPPORTS_PAGE_PARAMS = False` to use simple params

### **2. V2 Localization Paths** (`_load_localizations_via_relationship`)

```python
# Before (404 Not Found):
candidates = [
    f"/inAppPurchases/{inapp_id}/inAppPurchaseLocalizations",
]
if resource_type == "inAppPurchasesV2":
    candidates.insert(0, f"/inAppPurchasesV2/{inapp_id}/inAppPurchaseLocalizations")

# After (Works):
candidates = []
if resource_type == "inAppPurchasesV2":
    candidates.append(f"/v2/inAppPurchases/{inapp_id}/inAppPurchaseLocalizations")
candidates.append(f"/inAppPurchases/{inapp_id}/inAppPurchaseLocalizations")
```

### **3. V2 Price Paths** (`_fetch_inapp_prices`)

```python
# Before (404 Not Found):
paths = [f"/inAppPurchases/{inapp_id}/prices"]
if resource_type == "inAppPurchasesV2":
    paths.insert(0, f"/inAppPurchasesV2/{inapp_id}/inAppPurchasePrices")

# After (Works):
paths = []
if resource_type == "inAppPurchasesV2":
    # V2 uses pricePoints endpoint with include parameter
    paths.append((f"/v2/inAppPurchases/{inapp_id}/pricePoints", True))  # is_v2=True
paths.append((f"/inAppPurchases/{inapp_id}/prices", False))  # is_v2=False

# Request with include parameter for V2:
params = {"include": "priceTier,territory"} if is_v2 else None
response = _request("GET", path, params=params)

# Parse V2 response with included relationships:
included = response.get("included", [])
price_tier_map = _index_included(included, "inAppPurchasePriceTiers")
territory_map = _index_included(included, "territories")
parsed = _parse_price_point_entry(entry, price_tier_map, territory_map)
```

### **4. V2 Include Parameter** (`_get_inapp_purchase_snapshot`)

```python
# Before (400 Parameter Error):
params = {"include": "inAppPurchaseLocalizations,inAppPurchasePrices"}  # ❌ Invalid for V2

# After (Works):
if path.startswith("/v2/"):
    params = {"include": "inAppPurchaseLocalizations"}  # ✅ Valid for V2
else:
    params = {"include": "inAppPurchaseLocalizations,inAppPurchasePrices"}  # V1 only
```

### **4. V2 Localization Lookup** (`_load_localization_entries_v2`)

```python
# Before (Extra wrong path):
candidates = [
    f"/v2/inAppPurchases/{inapp_id}/inAppPurchaseLocalizations",
]
if resource_type == "inAppPurchasesV2":
    candidates.insert(0, f"/v2/inAppPurchasesV2/{inapp_id}/inAppPurchaseLocalizations")  # ❌ Wrong!

# After (Single correct path):
candidates = [f"/v2/inAppPurchases/{inapp_id}/inAppPurchaseLocalizations"]
```

### **5. Price Tier Response** (`list_price_tiers`)

```python
# Before (included proceeds):
tiers[tier_id] = {
    "tier": tier_id,
    "currency": attributes.get("currency"),
    "customerPrice": attributes.get("customerPrice"),
    "proceeds": attributes.get("proceeds"),  # Not available from appPricePoints
}

# After (customer price only):
tiers[tier_id] = {
    "tier": tier_id,
    "currency": attributes.get("currency"),
    "customerPrice": attributes.get("customerPrice"),
}
```

---

## Expected Behavior After Fixes

### **Price Tiers**
```
# Before:
[INFO] Apple API error 403: [FORBIDDEN_ERROR] The resource 'inAppPurchasePricePoints' has no allowed operations defined.
[INFO] Price points access forbidden with primary credentials. Switching to fallback credentials.
[ERROR] Fallback credentials also failed: Apple API 오류 403...

# After:
✅ Successfully fetched price tiers from /apps/{APP_ID}/appPricePoints
✅ Price tiers loaded: [Tier 1, Tier 2, Tier 3, ...]
```

### **V2 IAP Edit (Numeric ID like 6737119376)**
```
# Before:
[ERROR] Apple API error 404: [NOT_FOUND] The path provided does not match a defined resource type.
| URL: https://api.appstoreconnect.apple.com/v1/inAppPurchasesV2/6737119376/inAppPurchaseLocalizations
[ERROR] Apple API error 404: [NOT_FOUND] The path provided does not match a defined resource type.
| URL: https://api.appstoreconnect.apple.com/v1/inAppPurchasesV2/6737119376/inAppPurchasePrices

# After:
✅ Successfully fetched IAP detail from /v2/inAppPurchases/6737119376
✅ Successfully fetched localizations from /v2/inAppPurchases/6737119376/inAppPurchaseLocalizations
✅ Successfully fetched prices from /v2/inAppPurchases/6737119376/iapPriceSchedule
```

---

## Testing Recommendations

1. **Price Tiers:**
   - Navigate to Apple IAP section
   - Check if price tier dropdown loads
   - Verify tiers display with customer prices

2. **V1 IAP Edit (UUID):**
   - Select a V1 IAP (UUID like `a1b2c3d4-...`)
   - Click edit
   - Verify localizations and prices load correctly

3. **V2 IAP Edit (Numeric ID):**
   - Select a V2 IAP (numeric ID like `6737119376`)
   - Click edit
   - Verify localizations and prices load correctly
   - Check logs for no 404 errors

4. **Create New IAP:**
   - Try creating a new IAP
   - Select price tier from dropdown
   - Verify it saves correctly

---

## Files Modified

- **`apple_store.py`**
  - `_request_price_points()` - Changed to use `/apps/{APP_ID}/appPricePoints`
  - `_load_localizations_via_relationship()` - Fixed V2 localization paths
  - `_load_localization_entries_v2()` - Removed wrong path
  - `_fetch_inapp_prices()` - Fixed V2 price paths to use `/iapPriceSchedule`
  - `list_price_tiers()` - Removed `proceeds` field (not available from new endpoint)

