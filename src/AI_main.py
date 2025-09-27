from fastapi import FastAPI, Depends, HTTPException, UploadFile,  Query, Path, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import torch
from generate_story.lora_manager import get_lora_manager, ensure_model_loaded
from generate_story.generate_story import StoryBookGenerator
from generate_story.generate_image import ImageGenerator
from generate_story.generate_eval import StoryEvaluator
from generate_story.generate_summary import Summarizer
from pydantic import BaseModel
from sqlalchemy import or_
from sqlalchemy.orm import Session
from db.db_connector import SessionLocal
from db.db_models import FairyTale, FairyTaleImages
from datetime import date
import gc
import torch
import logging
import time
import json
import base64
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       
    allow_credentials=True,   
    allow_methods=["*"],        
    allow_headers=["*"]       
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# lora_manager 초기화
lora_manager = get_lora_manager()
while not ensure_model_loaded():
    time.sleep(1)
    print("베이스 모델 로딩 대기중...")

# 동화생성 모델 로드
sbg = StoryBookGenerator()

# 평가 모델 로드
story_evaluator = StoryEvaluator()

# 요약 모델 로드
summarizer = Summarizer()

# 스테이블 디퓨전 모델 로드
img_generator = ImageGenerator()
img_generator.load_diffusion_model()

class GenerateStoryRequest(BaseModel):
    uid: int
    type: int
    name: str
    age: int
    genre: str

class GenerateStoryResponse(BaseModel):
    message: str
    fid: int
        
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

# 동화 생성 API
@app.post("/generate_story")
def generate_story(req: GenerateStoryRequest, db: Session = Depends(get_db), stream: bool = Query(False)):    
    try:
        count = 1        
        while count <= 10:
            result = sbg.generate_story(name=req.name, age=req.age, genre=req.genre)
            eval_scores = story_evaluator.evaluate_single_story_fast(result['content'], result['prompt'])['scores']
            
            if all(score > 1 for score in eval_scores):
                break
            
            count += 1

        if count > 10:
            raise HTTPException(
            status_code=400,
            detail="10번 시도하였으나 유효한 동화를 생성하지 못했습니다."
        )        
        
        summary = summarizer.generate_summary(uid=req.uid, type=req.type, title=result['title'], contents=result["content"], max_new_tokens=200)["summary"]

        ft = FairyTale(
            uid=req.uid,
            type=2,
            title=result["title"],
            summary=summary,
            contents=' '.join(result["content"]).strip(),
            createDate=date.today(),
        )

        try:
            db.add(ft)
            db.commit()
            db.refresh(ft)
        except Exception as e:
            db.rollback()
            logging.error(f"Database error: {str(e)}", exc_info=True)

        story = db.query(FairyTale).filter(FairyTale.title == result['title']).first()

        if stream:
            def generate_pages():
                for i, page_content in enumerate(result["content"]):
                    page_summary = summarizer.generate_page_summaries([page_content])[0]
                    image_path, file_name = img_generator.generate_image(page_summary, result["title"])
                    
                    # 이미지 DB 저장
                    images = FairyTaleImages(fid=story.fid, image_path=image_path, file_name=file_name, createDate=date.today())
                    db.add(images)
                    db.commit()
                    
                    # 실제 이미지 파일 경로 구성
                    full_image_path = os.path.join(image_path, file_name)
                    
                    # 이미지를 base64로 인코딩
                    try:
                        with open(full_image_path, "rb") as image_file:
                            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                            image_data = f"data:image/png;base64,{image_base64}"
                    except Exception as e:
                        print(f"이미지 로딩 실패 ({full_image_path}): {e}")
                        image_data = None
                    
                    page_data = {
                        'page': i + 1, 
                        'content': page_content, 
                        'image': image_data,  # base64 형태로 전송
                        'total_pages': len(result["content"]), 
                        'title': result["title"] if i == 0 else None
                    }
                    yield f"data: {json.dumps(page_data, ensure_ascii=False)}\n\n"
                
                yield f"data: {json.dumps({'completed': True, 'fid': story.fid, 'message': '동화생성을 완료했습니다.'})}\n\n"
                
            return StreamingResponse(
                generate_pages(), 
                media_type="text/event-stream", 
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            # 기존 방식
            page_summaries = summarizer.generate_page_summaries(result["content"])
            for summary_text in page_summaries:
                image_path, file_name = img_generator.generate_image(summary_text, result["title"])
                images = FairyTaleImages(fid=story.fid, image_path=image_path, file_name=file_name, createDate=date.today())
                db.add(images)
                db.commit()
            
            return GenerateStoryResponse(message="동화생성을 완료했습니다.", fid=story.fid)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"동화 생성에 실패하였습니다.: {e}")