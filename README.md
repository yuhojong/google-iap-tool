# iap-manager

iap-manager는 Google Play 관리형 인앱 상품을 조회하고 생성할 수 있는 FastAPI 기반 도구입니다.

## 구성 요소
- **FastAPI**: 백엔드 API 제공
- **google-api-python-client**: Android Publisher API 연동
- **Vanilla HTML/CSS/JS**: 단일 페이지 사용자 인터페이스

## 사전 준비
1. Google Cloud에서 Android Publisher API가 활성화된 서비스 계정을 생성합니다.
2. 서비스 계정 키 JSON 파일을 다운로드하고 안전한 위치에 저장합니다.
3. 프로젝트 루트에 `.env` 파일을 생성하고 다음 값을 설정합니다.
   ```env
   PACKAGE_NAME=패키지.이름.예시
   GOOGLE_APPLICATION_CREDENTIALS=./service-account.json
   PRICE_TEMPLATES=[{"id":"tier_kr_us","label":"KRW 5500 / USD 4.99","default":{"currency":"KRW","price":"5500"},"regions":{"KR":{"currency":"KRW","price":"5500"},"US":{"currency":"USD","price":"4.99"}}}]
   ```
4. `.gitignore`에 의해 서비스 계정 키 파일은 커밋되지 않습니다.

## 실행 방법
1. 가상환경을 생성하고 활성화합니다.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. 의존성을 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```
3. 애플리케이션을 실행합니다.
   ```bash
   uvicorn main:app --reload
   ```
4. 브라우저에서 `http://localhost:8000`에 접속하여 UI를 확인합니다.

## Windows에서 원클릭 실행 스크립트 사용하기
회사 정책으로 PowerShell 스크립트 실행이 제한되어 있어도, 관리자 권한의 **명령 프롬프트(cmd.exe)**에서 `setup_and_run.cmd` 파일 하나로 같은 작업을 수행할 수 있습니다.

1. 프로젝트 루트에 `.env` 파일과 서비스 계정 키 JSON 파일을 준비합니다. `.env`에 상대 경로가 있다면 스크립트가 자동으로 절대 경로로 바꿔 줍니다.
2. 시작 메뉴에서 **명령 프롬프트**를 검색해 **관리자 권한으로 실행**합니다. (Python 설치 과정에서 UAC 확인 창이 나타날 수 있습니다.)
3. 프로젝트 디렉터리로 이동한 뒤 스크립트를 실행합니다.
   ```cmd
   cd C:\path\to\project
   setup_and_run.cmd
   ```

`setup_and_run.cmd`는 다음 작업을 순서대로 수행합니다.

- Python 3.11이 없으면 공식 설치 프로그램을 다운로드하여 사용자 영역에 조용히 설치합니다. (Windows 10/11 기본 `curl` 또는 PowerShell을 사용합니다.)
- 가상환경(`.venv`)을 생성하고 `requirements.txt`에 정의된 의존성을 모두 설치합니다.
- `.env` 파일을 읽어 필요한 환경 변수를 현재 세션과 사용자 환경 변수에 등록합니다.
- FastAPI 서버를 `http://localhost:8000`에서 실행합니다.

> **참고:** `.env` 파일이 없으면 환경 변수 등록 단계를 건너뜁니다. PowerShell 실행 정책이 허용된다면 기존 `setup_and_run.ps1` 스크립트를 그대로 사용할 수도 있습니다.

## 주요 API
- `GET /api/inapp/list?token=`: 관리형 인앱 상품 목록 및 페이지 토큰 반환
- `GET /api/pricing/templates`: `.env`에 정의된 가격 템플릿 목록 반환
- `POST /api/inapp/create`: 새 관리형 인앱 상품 생성 (SKU, 기본 언어, 번역, `price_template_id` 또는 직접 입력 가격)

## 가격 템플릿 구성
`PRICE_TEMPLATES` 환경 변수는 JSON 배열이어야 하며, 각 항목은 다음 필드를 포함합니다.

| 필드 | 설명 |
| --- | --- |
| `id` | 템플릿을 식별하는 고유 문자열 |
| `label` | UI 드롭다운에 표시될 이름 |
| `default` | `{ "currency": "KRW", "price": "5500" }` 형태의 기본 가격 정보 (`price` 대신 `priceMicros` 사용 가능) |
| `regions` | 지역 코드(`US`, `KR` 등)를 키로 하고 `{ "currency": "USD", "price": "4.99" }` 형식의 가격을 값으로 갖는 객체 |
| `description` *(선택)* | 템플릿 설명 문구 |

> **참고:** Google Play Console에서 이미 발행된 국가에는 모두 가격이 지정되어 있어야 API 호출이 성공합니다. 필요한 모든 지역 코드를 `regions`에 포함했는지 확인하세요.

## 주의 사항
- UI에서는 `.env`에 정의된 가격 템플릿만 선택할 수 있으며, 템플릿 정보는 서버 시작 시 로드됩니다.
- 서비스 계정 키 파일은 절대 저장소에 커밋하지 마세요.
- Android Publisher API 사용을 위해 필요한 권한이 서비스 계정에 부여되어야 합니다.
