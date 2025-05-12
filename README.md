# AI 김대리 - 드라이브 리포터

Google Drive 연동 보고서 생성 서비스의 MVP 버전입니다.

## 주요 기능

- Google Drive 폴더 내 PDF, TXT, Google Docs 파일 검색
- Cohere API를 활용한 문서 분석 및 요약
- Gradio 기반 사용자 인터페이스

## 설치 방법

1. Python 3.8 이상 설치
2. 의존성 패키지 설치:
```bash
pip install -r requirements.txt
```

3. 환경 변수 설정:
`.env` 파일을 생성하고 다음 변수들을 설정하세요:
```
COHERE_API_KEY=your_cohere_api_key
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
```

## 실행 방법

```bash
python main.py
```

## 프로젝트 구조

```
.
├── main.py              # 메인 애플리케이션 진입점
├── requirements.txt     # 프로젝트 의존성
├── src/
│   ├── api/            # API 관련 코드
│   ├── core/           # 핵심 비즈니스 로직
│   ├── services/       # 외부 서비스 연동
│   └── utils/          # 유틸리티 함수
└── tests/              # 테스트 코드
``` 
