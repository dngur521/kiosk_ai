import os
import requests
import torch
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from chromadb.utils import embedding_functions
from datetime import datetime

app = FastAPI()

# --- 1. 임베딩 모델 및 ChromaDB 설정 ---
print("⏳ iLog 임베딩 모델 로딩 중 (multilingual-e5-large)...")
embedding_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/multilingual-e5-large"
)

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="kiosk_menus",
    embedding_function=embedding_ef,
    metadata={"hnsw:space": "cosine"}
)

MENU_API_URL = "https://kemini-kiosk-api.duckdns.org/api/menu"

def refresh_menu_index():
    try:
        response = requests.get(MENU_API_URL)
        if response.status_code == 200:
            menu_data = response.json().get("data", [])
            ids, documents, metadatas = [], [], []

            for m in menu_data:
                ids.append(str(m["id"]))
                raw_ctx = m.get("semanticContext") or m.get("name")
                ctx = f"passage: {raw_ctx}"
                documents.append(ctx)
                metadatas.append({
                    "name": m["name"],
                    "description": m.get("description", "")
                })

            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            print(f"✅ ChromaDB 색인 완료: {len(menu_data)}개 메뉴 (Distance: Cosine)")
    except Exception as e:
        print(f"❌ 메뉴 색인 중 오류: {e}")

@app.on_event("startup")
async def startup_event():
    refresh_menu_index()

class QueryRequest(BaseModel):
    query: str

@app.post("/recommend")
async def get_recommendation(request: QueryRequest):
    # [디버깅] 요청 시간 및 쿼리 출력
    print(f"\n{'='*60}")
    print(f"🔍 [AI 추천 요청] {datetime.now().strftime('%H:%M:%S')}")
    print(f"💬 사용자 쿼리: \"{request.query}\"")
    print(f"{'-'*60}")

    query_text = f"query: {request.query}"
    
    results = collection.query(
        query_texts=[query_text],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    # 1. 점수 계산 및 가중치 적용
    temp_recommendations = []
    boost_keywords = ["상큼", "과일", "시원", "잠", "커피", "달콤", "디저트", "빵"]

    for i in range(len(results['ids'][0])):
        raw_score = 1 - results['distances'][0][i]
        name = results['metadatas'][0][i]['name']
        doc_content = results['documents'][0][i]
        
        applied_boosts = []
        final_score = raw_score
        for word in boost_keywords:
            if word in request.query and word in doc_content:
                final_score += 0.05
                applied_boosts.append(word)
        
        temp_recommendations.append({
            "id": results['ids'][0][i],
            "name": name,
            "description": results['metadatas'][0][i]['description'],
            "raw_score": raw_score,
            "score": round(final_score, 4),
            "boosts": applied_boosts
        })

    # 2. 최고점 기준 정렬 (순위 확정)
    temp_recommendations.sort(key=lambda x: x['score'], reverse=True)
    
    # 3. 최고점 확인 및 커트라인 계산 (백엔드 로직 동기화)
    max_score = temp_recommendations[0]['score'] if temp_recommendations else 0.0
    relative_threshold = max_score - 0.05
    min_absolute_threshold = 0.5 # 백엔드 기준

    print(f"📊 [판정 기준] 최고점: {max_score:.4f} / 상대 커트라인: {relative_threshold:.4f} (Min: {min_absolute_threshold})")
    print(f"{'-'*60}")

    # 4. 결과 출력 및 반환 데이터 구성
    for i, res in enumerate(temp_recommendations):
        # 🔥 백엔드 필터 로직: 점수가 0.5 이상 AND (최고점 - 0.05) 이상
        is_pass = res['score'] >= min_absolute_threshold and res['score'] >= relative_threshold
        status_icon = "✅ PASS" if is_pass else "❌ FAIL"
        
        boost_info = f" (+0.05 x {len(res['boosts'])} [{', '.join(res['boosts'])}])" if res['boosts'] else " (No Boost)"
        
        print(f"[{i+1}위] {res['name']} (ID: {res['id']})")
        print(f"   - Raw AI Score: {res['raw_score']:.4f}")
        print(f"   - Final Score:  {res['score']:.4f} {boost_info} -> {status_icon}")

    print(f"{'='*60}\n")
    return temp_recommendations

@app.post("/refresh")
async def refresh():
    refresh_menu_index()
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)