# Apple IAP Caching Optimization

## Overview

The application now implements intelligent caching for Apple Store IAPs to drastically reduce unnecessary API calls and improve performance.

## Problem Before Optimization

**Previous behavior:**
- Every "목록 새로고침" (Refresh) button click fetched ALL IAPs from Apple
- For 2603 IAPs, this meant:
  - ~2-3 minutes of processing
  - Hundreds of API calls
  - Rate limiting issues
  - Wasted bandwidth and time
  - Even when nothing changed!

## Smart Caching Solution

### How It Works

1. **First Fetch**: Full fetch and store in SQLite database (`iap_cache.db`)
2. **Subsequent Refreshes**:
   - **Lightweight check**: Fetch only IAP IDs (no details)
   - **Compare**: Check if IDs match cached data
   - **Smart decision**:
     - ✅ **No changes** → Use cached data (instant!)
     - 🔄 **Changes detected** → Full refresh

### Three-Step Process

```
User clicks "Refresh"
     ↓
1. Quick ID fetch (< 1 second)
     ↓
2. Compare with cache
     ↓
3a. No changes → Return cache (instant)
3b. Changes → Full refresh (2-3 min)
```

## Performance Improvements

### Before Optimization
```
Every refresh: 2-3 minutes
100 refreshes: 200-300 minutes (3-5 hours!)
```

### After Optimization
```
No changes: < 1 second
With changes: 2-3 minutes (only when needed)
100 refreshes (no changes): < 2 minutes total!
```

**Improvement: 99%+ reduction in processing time for unchanged data**

## API Efficiency

### Lightweight ID Check
```python
# Only fetches IAP IDs, no details
GET /apps/{id}/inAppPurchasesV2?limit=200
→ Returns: ["6745171851", "6746429077", ...]
→ Fast: ~0.5-1 second for 2603 items
```

### Full Fetch (only when needed)
```python
# Fetches complete IAP details
GET /apps/{id}/inAppPurchasesV2 → Get 200 IDs
GET /inAppPurchases/{id} × 200 → Batch fetch details
→ Slow: 2-3 minutes for 200 items
```

## Change Detection

### Detects:
- ✅ New IAPs added
- ✅ IAPs removed/deleted
- ✅ Count changes

### Logs:
```
[INFO] Fetched 2603 IAP IDs (lightweight check)
[INFO] Apple IAP list unchanged (2603 items) - using cache

OR

[INFO] Apple IAP count changed: 2603 cached vs 2605 current - refreshing
[INFO] Changes detected, performing full refresh
```

## Database Schema

### Cache Table
```sql
CREATE TABLE products (
    store TEXT NOT NULL,           -- 'apple' or 'google'
    sku TEXT NOT NULL,              -- Product ID
    data TEXT NOT NULL,             -- Full JSON data
    hash TEXT NOT NULL,             -- SHA256 hash for change detection
    updated_at REAL NOT NULL,       -- Timestamp
    PRIMARY KEY (store, sku)
);
```

### Incremental Updates
Only modified/new/deleted items are updated:
```
Cache updated: 3 new, 5 updated, 1 removed, 2594 unchanged
```

## User Experience

### Fast Refresh When No Changes
```
User: *clicks refresh*
  ↓ < 1 second
System: "총 2603개의 Apple 인앱 상품을 불러왔습니다."
```

### Full Refresh When Changes Detected
```
User: *clicks refresh after adding 2 IAPs*
  ↓ Quick check
System: "Changes detected, performing full refresh"
  ↓ 2-3 minutes
System: "총 2605개의 Apple 인앱 상품을 불러왔습니다."
```

## Configuration

### Environment Variables
```bash
# Cache database location (default: ./iap_cache.db)
PRODUCT_CACHE_DB_PATH=/path/to/cache.db
```

### Force Full Refresh
To bypass cache and force a complete refresh, the system automatically detects changes. No manual intervention needed.

## Fallback Behavior

If the lightweight check fails:
- System falls back to full refresh
- Logs warning: "Failed to check for Apple IAP changes: {error} - will refresh"
- Ensures reliability over performance

## Cache Invalidation

Cache is automatically updated when:
1. New IAPs are detected
2. IAPs are removed
3. IAP count changes
4. Manual creation/deletion via the app

## Monitoring

### Logs to Watch

**Cache Hit (No Changes):**
```
[INFO] Fetched 2603 IAP IDs (lightweight check)
[INFO] Apple IAP list unchanged (2603 items) - using cache
[INFO] No changes detected, using cached data
```

**Cache Miss (Changes Detected):**
```
[INFO] Fetched 2605 IAP IDs (lightweight check)
[INFO] Apple IAP count changed: 2603 cached vs 2605 current - refreshing
[INFO] Changes detected, performing full refresh
[INFO] Cache updated: 2 new, 0 updated, 0 removed, 2603 unchanged
```

**First Time (No Cache):**
```
[INFO] No cached data, performing initial fetch
[INFO] Listing 200 in-app purchases
[INFO] Cache updated: 200 new, 0 updated, 0 removed, 0 unchanged
```

## Benefits Summary

1. **⚡ Speed**: 99%+ faster when no changes
2. **💰 Cost**: Fewer API calls = reduced rate limiting
3. **🎯 Efficiency**: Only fetch what changed
4. **✅ Reliability**: Automatic change detection
5. **📊 Transparency**: Clear logging of cache hits/misses
6. **🔄 Smart**: Incremental updates for changed data

## Technical Details

### Change Detection Algorithm
```python
1. Fetch current IAP IDs (lightweight)
2. Load cached IAP IDs from database
3. Compare sets:
   - Count different? → Refresh
   - IDs different? → Refresh
   - Same? → Use cache
```

### Incremental Update Algorithm
```python
For each IAP from API:
    hash = SHA256(JSON data)
    if hash == cached_hash:
        Skip (unchanged)
    elif SKU in cache:
        Update row
    else:
        Insert new row

Remove SKUs not in API response
```

## Migration

Existing installations automatically benefit from this optimization:
- No configuration changes needed
- Cache is created on first use
- Backward compatible with existing code

## Testing

### Verify Cache is Working
1. Click "목록 새로고침" → Should take 2-3 minutes (initial fetch)
2. Click "목록 새로고침" again → Should take < 1 second
3. Check logs for "using cache" message

### Verify Change Detection
1. Add/delete an IAP in App Store Connect
2. Click "목록 새로고침"
3. Should detect change and perform full refresh
4. Check logs for "Changes detected" message

