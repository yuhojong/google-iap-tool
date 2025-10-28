@echo off
setlocal enabledelayedexpansion

REM 설정
set "APP_ID=6451133846"
set "PYTHON=python"
set "BASE=https://api.appstoreconnect.apple.com"
set "OUT=iap_all.json"

where jq >nul 2>nul || (echo [ERROR] jq.exe 필요. https://stedolan.github.io/jq/ 에서 받아 같은 폴더에 두세요.& exit /b 1)
where %PYTHON% >nul 2>nul || (echo [ERROR] python 없음 & exit /b 1)

REM JWT
for /f "usebackq delims=" %%A in (`%PYTHON% make_jwt.py 2^>^&1`) do set "JWT=%%A"
if not defined JWT (echo [ERROR] JWT 생성 실패 & exit /b 1)

REM 초기 URL (limit=200)
set "URL=%BASE%/v1/apps/%APP_ID%/inAppPurchasesV2?limit=200"

REM 빈 배열로 시작
echo {"data": []} > "%OUT%"

:LOOP
echo [*] GET "%URL%"
curl -s -H "Authorization: Bearer %JWT%" "%URL%" > resp.json

REM 401이면 토큰 재발급 1회 재시도
findstr /c:"NOT_AUTHORIZED" resp.json >nul && (
  echo [!] 401 -> refresh JWT
  for /f "usebackq delims=" %%A in (`%PYTHON% make_jwt.py 2^>^&1`) do set "JWT=%%A"
  curl -s -H "Authorization: Bearer %JWT%" "%URL%" > resp.json
)

REM 데이터 병합
jq -e . resp.json >nul 2>nul || (echo [ERROR] invalid JSON & exit /b 1)
jq -s "{data: (.[0].data + .[1].data)}" "%OUT%" resp.json > out_next.json
move /y out_next.json "%OUT%" >nul

REM 다음 링크 추출
for /f "usebackq delims=" %%N in (`jq -r ".links.next // empty" resp.json`) do set "NEXT=%%N"

if not defined NEXT goto DONE

REM same-URL 가드
if /I "%URL%"=="%NEXT%" (
  echo [WARN] links.next가 현재 URL과 동일 - 종료
  goto DONE
)

set "URL=%NEXT%"
set "NEXT="
goto LOOP

:DONE
echo [OK] Saved -> "%OUT%"
exit /b 0
