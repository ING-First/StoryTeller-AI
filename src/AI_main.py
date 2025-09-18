from fastapi import FastAPI, Depends, HTTPException, UploadFile,  Query, Path, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
import torch
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
@app.post("/generate_story", response_model=GenerateStoryResponse)
def generate_story(req: GenerateStoryRequest, db: Session = Depends(get_db)):
    try:
        count = 1        
        while count <= 10:
            sbg = StoryBookGenerator()
            sbg.load()
            result = sbg.generate_story(name=req.name, age=req.age, genre=req.genre)
            del sbg; gc.collect(); torch.cuda.empty_cache()
            
            eval = StoryEvaluator()
            eval_scores = eval.evaluate_single_story_fast(result['content'], result['prompt'])['scores']
            del eval; gc.collect(); torch.cuda.empty_cache()
            
            if all(score > 1 for score in eval_scores):
                break
            
            count += 1

        if count > 10:
            raise HTTPException(
            status_code=400,
            detail="10번 시도하였으나 유효한 동화를 생성하지 못했습니다."
        )        
        
        summarizer = Summarizer()
        summarizer.load_lora_model()
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

        page_summaries = summarizer.generate_page_summaries(result["content"])
        
        del summarizer; gc.collect(); torch.cuda.empty_cache()
        
        img_generator = ImageGenerator()
        img_generator.load_diffusion_model()
        
        for summary in page_summaries:
            image_path, file_name = img_generator.generate_image(summary, result["title"])

            images = FairyTaleImages(
                fid=story.fid,
                image_path=image_path,
                file_name=file_name,
                createDate=date.today(),
            )
            
            try:
                db.add(images)
                db.commit()
                db.refresh(images)
            except Exception as e:
                torch.cuda.empty_cache()
                print("이미지 데이터 저장 실패")
                
        del img_generator; gc.collect(); torch.cuda.empty_cache()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"동화 생성에 실패하였습니다.: {e}")
    
    finally:
        for obj_name in ["sbg", "summarizer", "img_generator", "eval"]:
            if obj_name in locals():
                try:
                    obj = locals()[obj_name]

                    # DiffusionPipeline GPU -> CPU 옮기기
                    if hasattr(obj, "pipe"):
                        try:
                            obj.pipe.to("cpu")
                        except:
                            pass

                    # Torch 모델 GPU -> CPU 옮기기
                    if hasattr(obj, "to"):
                        try:
                            obj.to("cpu")
                        except:
                            pass

                    # 원래 변수 자체를 해제
                    del locals()[obj_name]

                except Exception as e:
                    print(f"[WARN] {obj_name} 메모리 해제 중 오류 발생: {e}")

        gc.collect()
        torch.cuda.empty_cache()
        
    return GenerateStoryResponse(message="동화생성을 완료했습니다.", fid=story.fid)