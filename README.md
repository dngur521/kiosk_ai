# kiosk_ai

키오스크 시스템을 위한 AI 메뉴 추천 서버. 자연어 쿼리를 받아 의미 기반 검색으로 메뉴를 추천한다.

<img width="465" height="493" alt="SCR-20260429-bcgr" src="https://github.com/user-attachments/assets/6ea66316-c351-4c9a-a5f5-4442f63d1137" />
<img width="460" height="224" alt="SCR-20260429-bbsz" src="https://github.com/user-attachments/assets/afc5ceeb-5c6e-41ec-ba8f-e6c704fd9b92" />

## 동작 방식

1. 서버 시작 시 외부 메뉴 API에서 메뉴 목록을 가져와 ChromaDB에 임베딩 색인
2. `/recommend` 요청이 오면 `multilingual-e5-large` 모델로 코사인 유사도 검색
3. 키워드 부스트 및 점수 필터링 후 결과 반환

**필터 기준:** `score >= 0.5` AND `score >= (최고점 - 0.03)`

## 실행

```bash
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API

### `POST /recommend`

자연어 쿼리로 메뉴 추천 요청.

```json
// Request
{ "query": "달콤한 음료 추천해줘" }

// Response
[
  { "id": "1", "name": "카페라떼", "description": "...", "score": 0.82, "boosts": ["달콤"] },
  ...
]
```

### `POST /refresh`

메뉴 API에서 데이터를 다시 가져와 ChromaDB를 재색인한다. 메뉴가 변경됐을 때 호출.

```bash
curl -X POST http://localhost:8000/refresh
```

## 의존성

- `fastapi`, `uvicorn`
- `chromadb`
- `sentence-transformers` (`intfloat/multilingual-e5-large`)
- `torch`
