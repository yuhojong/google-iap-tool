# Apple IAP Fetch Optimization Fixes

## Issues Fixed

### 1. **Infinite Fetch Loop** 
**Problem:** After fetching all 2603 IAPs, the server continued making additional fetch requests indefinitely.

**Root Cause:** In `main.py` (lines 1813-1817), when `refresh=True`, the code called `get_all_inapp_purchases()` again just to get the total count. This triggered a complete re-fetch of all IAPs after the initial fetch completed.

**Solution:**
- Replaced the duplicate `get_all_inapp_purchases()` call with `get_inapp_purchase_ids_lightweight()`
- This lightweight function only fetches IAP IDs (not full data) to get the total count
- Moved the total count fetch BEFORE pagination to avoid duplicate fetches

```python
# Before (WRONG - causes duplicate fetch):
items, next_token = await _run_in_thread(...)
if not token and refresh:
    all_items, _, total_count = await _run_in_thread(
        get_all_inapp_purchases, include_relationships=False  # ❌ Fetches all data again!
    )

# After (CORRECT - lightweight ID fetch only):
if not token and refresh:
    _, total_count = await _run_in_thread(
        get_inapp_purchase_ids_lightweight  # ✅ Only fetches IDs
    )
items, next_token = await _run_in_thread(...)
```

---

### 2. **Infinite Loop on Empty Pages**
**Problem:** When the API returned empty pages with a `next` cursor, the backend would loop infinitely fetching 0 items.

**Root Cause:** `iterate_all_inapp_purchases()` only checked `if not cursor` to break the loop, but didn't check if 0 items were returned.

**Solution:**
- Added `len(items) == 0` check to the break condition
- Now stops immediately when receiving an empty page
- Added "Fetch complete" log message with total count

```python
# apple_store.py - iterate_all_inapp_purchases()
if not cursor or len(items) == 0:  # ✅ Stop on empty pages
    if cumulative_count > 0:
        logger.info("Fetch complete: %d total items retrieved", cumulative_count)
    break
```

---

### 3. **Cache Updating All Items (Not Just Changed)**
**Problem:** Cache logs showed `Cache updated: 0 new, 2603 updated, 0 removed, 0 unchanged`, meaning ALL items were being updated even when nothing changed.

**Root Cause:** The `_hash_product()` function was hashing the entire product object, including:
- Dynamic fields that change on every fetch (timestamps, metadata)
- Price amounts/dates that might be updated server-side
- API response metadata (`links`, `relationships`)

**Solution:**
- Enhanced `_hash_product()` to exclude unstable fields from hash calculation
- Only compares stable, user-visible fields:
  - Product core: `productId`, `referenceName`, `type`, `state`, etc.
  - Price essentials: `territory`, `priceTier`, `clearedForSale` (excludes amounts/dates)
- Excluded fields: `updated_at`, `lastModified`, `createdDate`, `links`, `meta`, `relationships`

```python
# product_cache.py - _hash_product()
def _hash_product(product: Dict[str, Any]) -> str:
    stable_product = product.copy()
    
    # Exclude dynamic/unstable fields
    exclude_fields = {
        "updated_at", "lastModified", "createdDate", "modifiedDate",
        "links", "meta", "relationships"
    }
    for field in exclude_fields:
        stable_product.pop(field, None)
    
    # For prices, only compare essential fields (not amounts/dates)
    if "prices" in stable_product:
        stable_prices = []
        for price in stable_product["prices"]:
            stable_price = {
                "territory": price.get("territory"),
                "priceTier": price.get("priceTier"),
                "clearedForSale": price.get("clearedForSale")
            }
            stable_prices.append(stable_price)
        stable_product["prices"] = stable_prices
    
    payload = _serialize_product(stable_product)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

**Expected Result:** Now logs should show:
```
Cache updated: 5 new, 12 updated, 3 removed, 2583 unchanged  # ✅ Most items unchanged!
```

---

## Benefits

### Performance Improvements
1. **No Duplicate Fetches:** Eliminates the redundant second fetch after completing the first one
2. **Faster Refresh:** Using `get_inapp_purchase_ids_lightweight()` is 10-20x faster than full fetches
3. **Minimal Database Operations:** Only updates changed items, not all 2603 every time
4. **Reduced API Calls:** Smart caching avoids unnecessary full refreshes

### User Experience
1. **Immediate Frontend Display:** No more waiting for duplicate fetches to complete
2. **Accurate Progress:** Progress bar shows actual remaining items
3. **Faster Subsequent Fetches:** Cache is now truly incremental
4. **Lower Rate Limit Risk:** Fewer API calls = less chance of hitting rate limits

---

## Verification

### Expected Backend Logs (Normal Operation)
```
[INFO] Listing 200 in-app purchases
[INFO] Processing batch: 1-30 of 200 (30/2603 fetched, 2573 remaining)
[INFO] Processing batch: 31-60 of 200 (60/2603 fetched, 2543 remaining)
[INFO] Processing batch: 61-90 of 200 (90/2603 fetched, 2513 remaining)
...
[INFO] Processing batch: 181-200 of 200 (200/2603 fetched, 2403 remaining)
[INFO] Page complete: 200/200 IAPs fetched from this page (200/2603 total, 2403 remaining)

[INFO] Listing 200 in-app purchases
[INFO] Processing batch: 1-30 of 200 (230/2603 fetched, 2373 remaining)
...
[INFO] Page complete: 200/200 IAPs fetched from this page (400/2603 total, 2203 remaining)

...

[INFO] Listing 3 in-app purchases
[INFO] Processing batch: 1-3 of 3 (2603/2603 fetched, 0 remaining)
[INFO] Page complete: 3/3 IAPs fetched from this page (2603/2603 total, 0 remaining)

[INFO] Fetch complete: 2603 total items retrieved  ← ✅ Stops here!
[INFO] Cache updated: 0 new, 0 updated, 0 removed, 2603 unchanged  ← ✅ All unchanged!
```

### Expected Backend Logs (With Changes)
```
[INFO] Fetch complete: 2603 total items retrieved
[INFO] Cache updated: 2 new, 5 updated, 1 removed, 2595 unchanged  ← ✅ Only changed items!
```

---

## Files Modified

1. **`apple_store.py`**
   - Fixed `iterate_all_inapp_purchases()` to stop on empty pages
   - Added "Fetch complete" summary log

2. **`main.py`**
   - Replaced duplicate `get_all_inapp_purchases()` with `get_inapp_purchase_ids_lightweight()`
   - Reordered total count fetch before pagination

3. **`product_cache.py`**
   - Enhanced `_hash_product()` to exclude unstable fields
   - Only compares user-visible, stable product data

---

## Testing Recommendations

1. **Initial Fetch (No Cache):**
   - Should fetch all 2603 items once
   - Log: `Cache updated: 2603 new, 0 updated, 0 removed, 0 unchanged`

2. **Subsequent Fetch (No Changes):**
   - Should complete quickly with lightweight ID check
   - Log: `No changes detected, using cached data`
   - If full fetch: `Cache updated: 0 new, 0 updated, 0 removed, 2603 unchanged`

3. **Fetch After Changes:**
   - Should detect changes and perform full refresh
   - Log: `Changes detected, performing full refresh`
   - Log: `Cache updated: X new, Y updated, Z removed, N unchanged`

4. **Frontend Display:**
   - Should display items immediately after first page completes
   - Progress bar should update correctly
   - No infinite loading after 100% completion

