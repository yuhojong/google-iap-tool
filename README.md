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

## 주요 API
- `GET /api/inapp/list?token=`: 관리형 인앱 상품 목록 및 페이지 토큰 반환
- `POST /api/inapp/create`: 새 관리형 인앱 상품 생성 (SKU, 제목, 설명, 원화 가격)

## 주의 사항
- 가격은 원화 기준으로 입력하며, API 호출 시 자동으로 마이크로 단위로 변환됩니다.
- 서비스 계정 키 파일은 절대 저장소에 커밋하지 마세요.
- Android Publisher API 사용을 위해 필요한 권한이 서비스 계정에 부여되어야 합니다.
