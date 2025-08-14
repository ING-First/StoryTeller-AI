from typing import Optional
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from db.db import SessionLocal
from db.db_models import FairyTale
from generate_summary import Summarizer
from datetime import date

app = FastAPI()

summarizer = Summarizer()
summarizer.load_lora_model()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
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