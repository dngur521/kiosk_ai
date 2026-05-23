# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals

(이 프로젝트엔 테스트 환경이 없으므로, 서버 실행 후 API 호출로 검증):

- "버그 수정" → "재현 쿼리를 명시하고, 서버 실행 후 `curl -X POST .../recommend`로 응답 확인"
- "점수 로직 변경" → "동일 쿼리로 변경 전후 score/boosts 값 비교"
- "리팩토링" → "변경 전후 `/recommend`, `/refresh` 응답이 동일한지 확인"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

## Running the server

```bash
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or directly:

```bash
python main.py
```

## Architecture

This is a single-file FastAPI app (`main.py`) that acts as an AI-powered menu recommendation backend for a kiosk system.

**Startup flow:** On startup, the app fetches all menus from the external API at `https://kemini-kiosk-api.duckdns.org/api/menu` and upserts them into an in-memory ChromaDB collection with cosine distance. The embedding model (`intfloat/multilingual-e5-large`) runs locally via SentenceTransformer. Each menu document is prefixed with `"passage: "` to match the E5 model's expected format.

**Recommendation flow (`POST /recommend`):**

1. Prefix the user query with `"query: "` (E5 model convention) and retrieve top-3 nearest neighbors from ChromaDB.
2. Convert cosine distance to similarity score: `score = 1 - distance`.
3. Apply boosts: +0.20 if the query string appears in the menu name; +0.05 per keyword from `boost_keywords` that appears in both the query and the menu's document.
4. Sort by final score descending. Filter: keep only items where `score >= 0.5` AND `score >= (max_score - 0.03)`. This threshold mirrors the backend kiosk logic.
5. Return all candidates (both passing and failing) — the frontend/backend is expected to apply the same filter logic.

**`POST /refresh`** — re-fetches and re-indexes menus from the external API without restarting the server.

## Key design details

- ChromaDB is in-memory (no persistence); the menu index is lost on restart and rebuilt from the live API.
- The `semanticContext` field from the API is preferred over `name` for the indexed document text.
- Filtering thresholds (`0.5` absolute, `max - 0.03` relative) are intentionally kept in sync with the downstream kiosk backend.

## Commit Message Convention

Format: `{emoji} {type}: {description (한국어, 한 줄)`

| Type       | Emoji | 용도                     |
| ---------- | ----- | ------------------------ |
| `feat`     | ✨    | 새 기능                  |
| `fix`      | 🐛    | 버그 수정                |
| `docs`     | 📝    | 문서 작성·수정           |
| `refactor` | ♻️    | 기능 변경 없는 코드 개선 |
| `chore`    | 🔧    | 빌드·설정·의존성 변경    |

Example: `✨ feat: 음성 주문 취소 기능 추가`
