# CSV Import Resilience Fix

## Problem
When importing Google Play IAPs via CSV, if even one IAP fails to create (e.g., due to missing regional prices), the entire import fails and stops.

**Previous behavior:**
- ❌ Import stops on first error
- ❌ All remaining IAPs are not processed
- ❌ No visibility into which items failed and why
- ❌ User has to fix CSV and re-import everything

**Example:**
```
Row 1-4: ✅ Created successfully
Row 5: ❌ Failed (regional pricing issue)
Row 6-77: ❌ Never processed!
```

---

## Solution
**Continue processing despite individual failures** and provide detailed error reporting:

### Backend Changes

#### 1. Import Apply Endpoint (`main.py`)
- Wrapped each operation in try-except
- Track failed items with row number, SKU, action, and error message
- Return detailed summary including failed items
- Log each failure for debugging

```python
@app.post("/api/google/inapp/import/apply")
async def api_import_apply(request: ImportApplyRequest):
    results = {"create": 0, "update": 0, "delete": 0, "failed": 0}
    failed_items = []

    for idx, op in enumerate(request.operations, start=1):
        try:
            # Process operation
            ...
        except Exception as exc:
            # Log and continue
            failed_items.append({
                "row": idx,
                "sku": ...,
                "action": op.action,
                "error": str(exc)
            })
            results["failed"] += 1
            continue
    
    response_data = {"status": "ok", "summary": results}
    if failed_items:
        response_data["failed_items"] = failed_items
    return response_data
```

#### 2. Bulk Create Apply Endpoint (`main.py`)
- Same error handling approach for "new import" feature
- Track failed items during bulk creation
- Return detailed error information

### Frontend Changes (`static/index.html`)

#### 1. Import Apply Handler
- Parse response to check for `failed_items`
- Display detailed failure information
- Show success summary alongside failures

```javascript
const result = await res.json();

if (result.failed_items && result.failed_items.length > 0) {
    const failedSummary = result.failed_items.map(item => 
        `행 ${item.row}: ${item.sku} (${item.error})`
    ).join('\n');
    importStatusEl.textContent = `일부 항목 적용 실패:\n${failedSummary}\n\n성공: ${result.summary.create}개 생성`;
} else {
    importStatusEl.textContent = '변경 사항이 적용되었습니다.';
}
```

#### 2. New Import Handler
- Same error handling for bulk create
- Display detailed failure information
- Show which rows failed and why

---

## How It Works

### Example Scenario

**CSV Import with 10 IAPs:**

```
Row 1: gem_100  ✅ Created
Row 2: gem_200  ✅ Created  
Row 3: gem_300  ❌ Failed (regional pricing)
Row 4: gem_400  ✅ Created
Row 5: gem_500  ✅ Created
Row 6: gem_600  ❌ Failed (duplicate SKU)
Row 7: gem_700  ✅ Created
Row 8: gem_800  ✅ Created
Row 9: gem_900  ✅ Created
Row 10: gem_1000 ✅ Created
```

**Result:**
```json
{
  "status": "ok",
  "summary": {
    "create": 8,
    "update": 0,
    "delete": 0,
    "failed": 2
  },
  "failed_items": [
    {
      "row": 3,
      "sku": "gem_300",
      "action": "create",
      "error": "Must provide a price for each region"
    },
    {
      "row": 6,
      "sku": "gem_600",
      "action": "create",
      "error": "SKU already exists"
    }
  ]
}
```

**Frontend Display:**
```
일부 항목 적용 실패:
행 3: gem_300 (Must provide a price for each region)
행 6: gem_600 (SKU already exists)

성공: 8개 생성, 0개 수정, 0개 삭제
```

---

## Benefits

| Before | After |
|--------|-------|
| ❌ Import stops on first error | ✅ Continues processing all items |
| ❌ All remaining items fail | ✅ Processes all valid items |
| ❌ No visibility into failures | ✅ Detailed failure report |
| ❌ Must fix CSV and re-import | ✅ Fix only failed items |
| ❌ Wasteful: Re-create successful items | ✅ Efficient: Only create missing items |

---

## Error Types Handled

### 1. Validation Errors (HTTPException)
- **Behavior:** Re-raised (stops all processing)
- **Reason:** Indicates structural problem with request
- **Examples:** Missing required fields, invalid action type

### 2. API Errors (Exception)
- **Behavior:** Logged, skipped, continue
- **Reason:** Item-specific problem doesn't affect other items
- **Examples:** Regional pricing, duplicate SKU, permission issues

### 3. Network/Timeout Errors
- **Behavior:** Logged, skipped, continue
- **Reason:** Individual item failure shouldn't block others
- **Examples:** API timeout, connection error

---

## Testing

### Test Case 1: Mixed Success/Failures
**CSV:** 10 rows, 2 invalid SKUs, 1 with missing regional prices
**Expected:** 7 successful, 3 failed, detailed error report

### Test Case 2: All Failures
**CSV:** 10 rows, all duplicate SKUs
**Expected:** 0 successful, 10 failed, detailed error report

### Test Case 3: All Success
**CSV:** 10 rows, all valid
**Expected:** 10 successful, 0 failed, "처리 완료" message

### Test Case 4: API Down
**Scenario:** Google API temporarily unavailable mid-import
**Expected:** Processes items until failure, logs errors, returns partial results

---

## Files Modified

- **`main.py`** - Updated `api_import_apply()` and `api_bulk_create_apply()` with error tracking
- **`static/index.html`** - Updated import status display to show failed items

---

## Summary

The CSV import now gracefully handles individual item failures, continuing to process remaining items and providing detailed error reports. Users can:

1. ✅ See which items failed and why
2. ✅ Fix only the failed items in CSV
3. ✅ Re-import only failed items
4. ✅ Avoid re-processing successful items
5. ✅ Debug issues with detailed error messages

This makes the import process **robust**, **efficient**, and **user-friendly**! 🎉

