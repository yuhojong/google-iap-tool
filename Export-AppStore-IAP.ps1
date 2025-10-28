param(
  [Parameter(Mandatory=$true)]
  [string]$AppId,                      # 숫자 Apple ID (예: 6451133846)
  [string]$PythonExe = "python",       # 필요하면 python 경로 지정
  [int]$JwtTtlSec = 1200               # JWT 유효기간 (기본 20분)
)

# -------- Settings --------
$BaseUrl = "https://api.appstoreconnect.apple.com"
$TimeStamp = (Get-Date).ToString("yyyyMMdd_HHmmss")
$OutJson  = "iap_export_$TimeStamp.json"
$OutIapCsv = "iap_list_$TimeStamp.csv"
$OutLocCsv = "iap_localizations_$TimeStamp.csv"

# -------- JWT Helper --------
$GLOBAL:JwtIssuedAt = 0
$GLOBAL:Jwt = $null

function Get-JWT {
  $now = [int][double]::Parse((Get-Date -UFormat %s))
  if ($GLOBAL:Jwt -and ($now - $GLOBAL:JwtIssuedAt) -lt ($JwtTtlSec - 120)) {
    return $GLOBAL:Jwt  # 아직 충분히 유효
  }
  Write-Host "Generating JWT via $PythonExe make_jwt.py ..."
  $token = & $PythonExe "make_jwt.py" 2>$null
  if (-not $token) { throw "Failed to get JWT from make_jwt.py" }
  $GLOBAL:Jwt = $token.Trim()
  $GLOBAL:JwtIssuedAt = $now
  return $GLOBAL:Jwt
}

function Invoke-Asc([string]$Uri) {
  $jwt = Get-JWT
  try {
    return Invoke-RestMethod -Headers @{ Authorization = "Bearer $jwt" } -Uri $Uri -Method GET
  } catch {
    # 만료/시계오차 등으로 401일 수 있으니 한 번 재시도
    Start-Sleep -Milliseconds 400
    $jwt = Get-JWT
    return Invoke-RestMethod -Headers @{ Authorization = "Bearer $jwt" } -Uri $Uri -Method GET
  }
}

function Get-Paged([string]$Uri) {
  $all = @()
  $next = $Uri
  do {
    $res = Invoke-Asc $next
    if ($res.data) { $all += $res.data }
    $next = $null
    if ($res.links -and $res.links.next) { $next = $res.links.next }
  } while ($next)
  return ,$all
}

# -------- 1) 앱의 모든 IAP(V2) 목록 수집 --------
Write-Host "Fetching IAP list for app $AppId ..."
$iapUri = "$BaseUrl/v1/apps/$AppId/inAppPurchasesV2?limit=200"
$iapList = Get-Paged -Uri $iapUri

if (-not $iapList -or $iapList.Count -eq 0) {
  Write-Warning "No IAPs found for app $AppId."
}

# -------- 2) 각 IAP 상세 + 현지화 수집 --------
$report = @()
$flatIap = @()
$flatLoc = @()

foreach ($iap in $iapList) {
  $iapId   = $iap.id
  $attrs   = $iap.attributes
  $refName = $attrs.referenceName
  Write-Host "IAP: $refName ($iapId)" -ForegroundColor Cyan

  # 상세 (v2/inAppPurchases/{id})
  $detailUri = "$BaseUrl/v2/inAppPurchases/$iapId"
  $detail = Invoke-Asc $detailUri

  # 현지화 목록 (name/locale/description만 필드 제한)
  $locUri = "$BaseUrl/v2/inAppPurchases/$iapId/inAppPurchaseLocalizations?limit=200&fields[inAppPurchaseLocalizations]=name,locale,description"
  $locs = Get-Paged -Uri $locUri

  # 리포트 구조 (JSON 저장용)
  $report += [pscustomobject]@{
    iapId = $iapId
    attributes = $detail.data.attributes
    relationships = $detail.data.relationships
    localizations = $locs
  }

  # IAP 요약 (CSV용)
  $flatIap += [pscustomobject]@{
    iapId           = $iapId
    referenceName   = $attrs.referenceName
    productId       = $attrs.productId
    inAppType       = $attrs.inAppPurchaseType
    state           = $attrs.state
    familySharable  = $attrs.familySharable
  }

  # 현지화 평탄화 (CSV용)
  foreach ($loc in $locs) {
    $a = $loc.attributes
    $flatLoc += [pscustomobject]@{
      iapId       = $iapId
      locale      = $a.locale
      name        = $a.name
      description = $a.description
    }
  }
}

# -------- 3) 파일 저장 --------
# (1) 전체 JSON
$report | ConvertTo-Json -Depth 100 | Out-File -FilePath $OutJson -Encoding utf8
Write-Host "Saved JSON -> $OutJson" -ForegroundColor Green

# (2) IAP 요약 CSV
$flatIap | Export-Csv -Path $OutIapCsv -NoTypeInformation -Encoding UTF8
Write-Host "Saved IAP CSV -> $OutIapCsv" -ForegroundColor Green

# (3) 현지화 CSV
$flatLoc | Export-Csv -Path $OutLocCsv -NoTypeInformation -Encoding UTF8
Write-Host "Saved Localizations CSV -> $OutLocCsv" -ForegroundColor Green
