from typing import Union, Optional
import os, requests
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from db import SessionLocal
from db_models import FairyTale
from generate_summary import Summarizer
from datetime import date
from dotenv import load_dotenv


load_dotenv()
app = FastAPI()

REMOTE_GENERATE_URL = os.getenv("REMOTE_GENERATE_URL")
REQUEST_TIMEOUT = float(os.getenv("REMOTE_TIMEOUT", "5"))

summarizer = Summarizer()
summarizer.load_lora_model()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class GenerateStoryRequest(BaseModel):
    uid: int
    type: int
    name: str
    age: int
    genre: str

class GenerateStoryResponse(BaseModel):
    uid: int
    type: int
    title: str
    contents: str
        
class GenerateRequest(BaseModel):
    uid: int
    type: int
    title: str
    contents: str
    
class GenerateResponse(BaseModel):
    uid: int
    type: int
    title: str
    summary: str
    contents: str
    createDate: date

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}


@app.post("/generate", response_model=GenerateStoryResponse)
def generate(req: GenerateStoryRequest):
    try:
        resp = requests.post(
            REMOTE_GENERATE_URL,
            json=req.dict(),
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"upstream_unreachable: {e}")

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"upstream_error: {resp.text}")

    data = resp.json() 
    return GenerateStoryResponse(**data)


@app.post("/summarize", response_model=GenerateResponse)
def create_summarization(req: GenerateRequest, db: Session = Depends(get_db)):
    if req.type not in (1, 2):
        raise HTTPException(status_code=400, detail="type은 1 혹은 2로 입력해야 합니다.")
      
    try:
        tale_data = summarizer.generate_summary(
            uid=req.uid, type=req.type, title=req.title, contents=req.contents, max_new_tokens=100
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generation_failed: {e}")
    
    if not tale_data.get("success", True):
        raise HTTPException(status_code=500, detail=tale_data.get("summary", "요약 생성에 실패하였습니다."))
    
    ft = FairyTale(
        uid=tale_data["uid"],
        type=tale_data["type"],
        title=tale_data["title"],
        summary=tale_data["summary"],
        contents=tale_data["contents"],
        createDate=tale_data["createDate"],
    )
      
    try:
      db.add(ft)
      db.flush()
      db.refresh(ft)
      db.commit()
    except Exception as e:
      db.rollback()
      raise HTTPException(status_code=500, detail=f"db_error: {e}")
    
    return {
        "uid": ft.uid,
        "type": ft.type,
        "title": ft.title,
        "summary": ft.summary,
        "contents": ft.contents,
        "createDate": ft.createDate,
    }