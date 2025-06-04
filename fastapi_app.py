import json
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.dataset import MetaQA
from src.indexer import Indexer
from src.retriever import Retriever
from src.llm import LLM

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.on_event("startup")
async def startup_event():
    global configs, dataset, retriever, llm
    configs = json.load(open('configs/azure_fastapi.json'))
    dataset = MetaQA(configs)
    KG = dataset.get_KG()
    indexer = Indexer(configs)
    indexer.build_index(KG)
    retriever = Retriever(configs, KG)
    llm = LLM(configs)

@app.post("/query")
async def query(req: QueryRequest):
    try:
        query_graph = llm.chat(json.dumps({"task": "rewrite", "question": req.question}))
        query_graph = json.loads(query_graph)
        result = retriever.retrieve(query_graph)
        evidence = [e[1] for e in result['results']]
        answer_prompt = json.dumps({"task": "answer", "question": req.question, "evidence": evidence})
        answer = llm.chat(answer_prompt)
        return {"answer": answer, "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
