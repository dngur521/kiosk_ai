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

    recommendations = []
    boost_keywords = ["상큼", "과일", "시원", "잠", "커피", "달콤", "디저트", "빵"]

    for i in range(len(results['ids'][0])):
        # 순수 AI 점수 (AI Raw Score)
        raw_score = 1 - results['distances'][0][i]
        
        name = results['metadatas'][0][i]['name']
        doc_content = results['documents'][0][i]
        
        # 가중치 계산 과정 추적
        applied_boosts = []
        final_score = raw_score
        
        for word in boost_keywords:
            if word in request.query and word in doc_content:
                final_score += 0.05
                applied_boosts.append(word)
        
        # [디버깅] 각 메뉴별 상세 로그
        boost_info = f" (+0.05 x {len(applied_boosts)} [{', '.join(applied_boosts)}])" if applied_boosts else " (No Boost)"
        status_icon = "✅ PASS" if final_score >= 0.75 else "❌ FAIL"
        
        print(f"[{i+1}위] {name} (ID: {results['ids'][0][i]})")
        print(f"   - Raw AI Score: {raw_score:.4f}")
        print(f"   - Final Score:  {final_score:.4f} {boost_info} -> {status_icon}")
        
        recommendations.append({
            "id": results['ids'][0][i],
            "name": name,
            "description": results['metadatas'][0][i]['description'],
            "score": round(final_score, 4)
        })

    recommendations.sort(key=lambda x: x['score'], reverse=True)
    print(f"{'='*60}\n")

    return {"recommendations": recommendations}

@app.post("/refresh")
async def refresh():
    refresh_menu_index()
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)