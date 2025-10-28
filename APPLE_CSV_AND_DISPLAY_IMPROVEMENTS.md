# Apple IAP CSV Management and Display Improvements

## Overview
This document outlines the implementation of CSV bulk management for Apple IAPs (similar to Google Play) and fixes for display issues in the IAP list.

---

## Tasks

### 1. CSV Bulk Management
**Goal:** Allow users to export all Apple IAPs to CSV, edit them, and import changes back.

**Features:**
- Export all IAPs to CSV with all fields
- Import CSV to preview changes (create, update, delete)
- Apply changes in bulk
- Progress tracking during import

### 2. Display Fixes
**Goal:** Fix data display issues in the Apple IAP list table.

**Issues to fix:**
- "참조 이름" (Reference Name) column: Should display Korean product name from localizations
- "가격 티어" (Price Tier) column: Should display Korean price from appPricePoints API

### 3. Price Tier Selection
**Goal:** Add proper price tier selection with Korean prices.

**Feature:**
- Fetch price tiers from `/apps/{APP_ID}/appPricePoints` with KOR territory
- Display Korean prices in dropdown
- Allow users to select appropriate tier when creating/editing IAPs

---

## Implementation Plan

### Phase 1: Backend API Endpoints

#### 1.1 Apple IAP CSV Export (`/api/apple/inapp/export`)
```python
@app.get("/api/apple/inapp/export")
@csv_processing_endpoint
async def api_apple_export():
    """Export all Apple IAPs to CSV."""
    # Get all IAPs with full details
    items, _ = await _run_in_thread(get_all_inapp_purchases, include_relationships=True)
    
    # Determine locales from all products
    locales = _collect_locales_from_apple_products(items)
    
    # Build CSV with columns:
    # - productId, referenceName, type, state, clearedForSale, familySharable
    # - priceTier, territory
    # - name_en-US, description_en-US, name_ko, description_ko, etc.
    
    return StreamingResponse(csv_content, media_type="text/csv")
```

#### 1.2 Apple IAP CSV Import Preview (`/api/apple/inapp/import/preview`)
```python
@app.post("/api/apple/inapp/import/preview")
@csv_processing_endpoint
async def api_apple_import_preview(file: UploadFile):
    """Preview Apple IAP changes from CSV."""
    # Parse CSV
    # Compare with existing IAPs
    # Return operations: {create: [], update: [], delete: []}
    return {"locales": [...], "operations": [...], "summary": {...}}
```

#### 1.3 Apple IAP CSV Import Apply (`/api/apple/inapp/import/apply`)
```python
@app.post("/api/apple/inapp/import/apply")
async def api_apple_import_apply(request: Request):
    """Apply Apple IAP changes from CSV preview."""
    # Execute create/update/delete operations
    # Return success/failure counts
    return {"status": "ok", "summary": {...}, "errors": [...]}
```

### Phase 2: Frontend UI

#### 2.1 Add CSV Management Section
Add new menu item and section to Apple store navigation:
```html
<nav class="menu" data-store="apple">
    <button type="button" class="menu-button is-active" 
            data-target="apple-section-product-list">인앱 상품 목록</button>
    <button type="button" class="menu-button" 
            data-target="apple-section-bulk-management">CSV 벌크 관리</button>
    <button type="button" class="menu-button" 
            data-target="apple-section-form">인앱 상품 등록/수정</button>
</nav>

<section id="apple-section-bulk-management" class="app-section" hidden>
    <h2>Apple IAP 벌크 관리</h2>
    <div class="csv-actions">
        <button id="apple-export-btn">CSV 내보내기</button>
        <form id="apple-import-form">
            <input type="file" id="apple-import-file" accept=".csv" />
            <button type="button" id="apple-import-preview-btn">미리보기</button>
            <button type="button" id="apple-import-apply-btn" disabled>적용</button>
        </form>
    </div>
    <div id="apple-import-status" class="status"></div>
    <div id="apple-import-preview"></div>
</section>
```

#### 2.2 CSV Export Handler
```javascript
async function handleAppleExport() {
    const res = await fetch('/api/apple/inapp/export');
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `apple-iap-${timestamp}.csv`;
    anchor.click();
}
```

#### 2.3 CSV Import Handlers
```javascript
async function handleAppleImportPreview() {
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    const res = await fetch('/api/apple/inapp/import/preview', {
        method: 'POST',
        body: formData
    });
    
    const data = await res.json();
    renderAppleImportPreview(data);
}

async function handleAppleImportApply() {
    const res = await fetch('/api/apple/inapp/import/apply', {
        method: 'POST',
        body: JSON.stringify(previewData)
    });
    
    const result = await res.json();
    showAppleImportResult(result);
}
```

### Phase 3: Display Fixes

#### 3.1 Fix "참조 이름" Column
**Current:** Shows `referenceName` (internal reference)
**Should show:** Korean product name from `localizations['ko']` or first available locale

```javascript
// In renderAppleTable():
const displayName = item.localizations?.['ko']?.name || 
                    item.localizations?.['ko-KR']?.name ||
                    Object.values(item.localizations || {})[0]?.name ||
                    item.referenceName ||
                    '';
```

#### 3.2 Fix "가격 티어" Column
**Current:** Shows tier ID (e.g., "1", "2")
**Should show:** Korean price (e.g., "₩1,200")

**Solution:** Fetch price from appPricePoints and cache:
```javascript
// Load Korean prices for all tiers
const priceMap = await loadApplePriceMap('KOR');

// In renderAppleTable():
const tier = item.prices?.[0]?.priceTier || '';
const korPrice = priceMap[tier] || tier;
```

#### 3.3 Add Price Tier Dropdown with Korean Prices
**Current:** Simple text input or select with tier IDs
**Should be:** Dropdown with formatted Korean prices

```javascript
// Load price tiers with Korean prices
async function loadApplePriceTiersWithPrices(territory = 'KOR') {
    const res = await fetch(`/api/apple/pricing/tiers?territory=${territory}`);
    const tiers = await res.json();
    
    // Populate dropdown
    tiers.forEach(tier => {
        const option = document.createElement('option');
        option.value = tier.tier;
        option.textContent = `Tier ${tier.tier}: ${formatKoreanPrice(tier.customerPrice)}`;
        select.appendChild(option);
    });
}
```

---

## CSV Format

### Columns
1. **Core Fields:**
   - `productId` (required, unique)
   - `referenceName` (required)
   - `type` (CONSUMABLE, NON_CONSUMABLE, NON_RENEWING_SUBSCRIPTION)
   - `state` (READY_TO_SUBMIT, WAITING_FOR_REVIEW, etc.)
   - `clearedForSale` (TRUE/FALSE)
   - `familySharable` (TRUE/FALSE)
   - `reviewNote` (optional)

2. **Pricing:**
   - `priceTier` (1, 2, 3, etc.)
   - `territory` (KOR, USA, etc.)

3. **Localizations:** (dynamic based on configured locales)
   - `name_ko`, `description_ko`
   - `name_en-US`, `description_en-US`
   - `name_ja`, `description_ja`
   - etc.

### Example CSV
```csv
productId,referenceName,type,state,clearedForSale,familySharable,priceTier,territory,name_ko,description_ko,name_en-US,description_en-US
ios_gem_100,100 Gems,CONSUMABLE,APPROVED,TRUE,FALSE,1,KOR,보석 100개,게임 내에서 사용할 수 있는 보석 100개,100 Gems,100 gems for use in the game
ios_premium,Premium Upgrade,NON_CONSUMABLE,APPROVED,TRUE,TRUE,10,KOR,프리미엄 업그레이드,모든 프리미엄 기능 잠금 해제,Premium Upgrade,Unlock all premium features
```

---

## API Response Examples

### Export Response
```
Content-Type: text/csv
Content-Disposition: attachment; filename="apple-iap-20250428120000.csv"

productId,referenceName,type,state,...
ios_gem_100,100 Gems,CONSUMABLE,APPROVED,...
ios_gem_500,500 Gems,CONSUMABLE,APPROVED,...
```

### Import Preview Response
```json
{
  "locales": ["ko", "en-US", "ja"],
  "operations": [
    {
      "action": "update",
      "productId": "ios_gem_100",
      "changes": {
        "priceTier": {"old": "1", "new": "2"},
        "name_ko": {"old": "보석 100개", "new": "보석 100개 (특별 할인)"}
      }
    },
    {
      "action": "create",
      "productId": "ios_gem_1000",
      "data": {...}
    },
    {
      "action": "delete",
      "productId": "ios_old_product",
      "reason": "Not in CSV"
    }
  ],
  "summary": {
    "create": 1,
    "update": 1,
    "delete": 1
  }
}
```

### Import Apply Response
```json
{
  "status": "ok",
  "summary": {
    "create": 1,
    "update": 1,
    "delete": 1,
    "failed": 0
  },
  "errors": []
}
```

---

## Files to Modify

### Backend
- `main.py`: Add 3 new endpoints (export, import preview, import apply)
- `apple_store.py`: No changes needed (existing functions are sufficient)

### Frontend
- `static/index.html`:
  - Add new section `apple-section-bulk-management`
  - Add menu button for CSV management
  - Add CSV export/import handlers
  - Fix display logic for 참조 이름 and 가격 티어
  - Add price tier dropdown with Korean prices
  - Update `renderAppleTable()` to show correct data
  - Update form to load and display Korean prices

---

## Testing Checklist

### CSV Export
- [ ] Export downloads CSV file
- [ ] CSV contains all IAPs
- [ ] CSV contains all localizations
- [ ] CSV is properly formatted UTF-8 with BOM

### CSV Import
- [ ] Preview correctly detects creates/updates/deletes
- [ ] Preview shows diff for changes
- [ ] Apply executes all operations
- [ ] Errors are handled gracefully
- [ ] Progress is shown during import

### Display
- [ ] "참조 이름" shows Korean product name
- [ ] "가격 티어" shows Korean price (₩X,XXX)
- [ ] Price dropdown shows all tiers with Korean prices
- [ ] Form correctly saves selected price tier

---

## Next Steps

1. ✅ Create this implementation document
2. ⏳ Implement backend CSV export endpoint
3. ⏳ Implement backend CSV import endpoints
4. ⏳ Add frontend CSV management UI
5. ⏳ Fix display issues in IAP list
6. ⏳ Add Korean price tier selection
7. ⏳ Test all functionality

