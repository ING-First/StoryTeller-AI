from typing import Union, Optional, List
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from fastapi import FastAPI, Depends, HTTPException, UploadFile,  Query, Path, File, Form, Body
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import text, or_
from sqlalchemy.orm import Session
from db.db_connector import SessionLocal
from db.db_models import Users, FairyTale, FairyTaleLog, Voices, FairyTaleImages
from generate_story.generate_summary import Summarizer
from datetime import date, datetime, timedelta
from passlib.context import CryptContext
from jose import jwt
from dotenv import load_dotenv
from generate_story.generate_story import StoryBookGenerator
from generate_story.generate_sound import SoundGenerator
from generate_story.generate_image import ImageGenerator
from generate_story.story_reading import StoryReader
from generate_story.generate_eval import StoryEvaluator
import os
import requests
import re
import httpx
import gc
import torch
import logging
import base64
import glob

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       
    allow_credentials=True,   
    allow_methods=["*"],        
    allow_headers=["*"]       
)


sg = SoundGenerator()
reader = StoryReader()

load_dotenv(override=False)

security = HTTPBearer()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))

XI_API_KEY = os.getenv("XI_API_KEY") or os.getenv("API_KEY")
ELEVEN_ADD_URL = "https://api.elevenlabs.io/v1/voices/add"

pattern = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*()_\-+=\[\]{}\\|;:\'",.<>/?`~])[A-Za-z\d!@#$%^&*()_\-+=\[\]{}\\|;:\'",.<>/?`~]{8,15}$')

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return True
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="토큰이 만료되었습니다.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")
    

class VoiceRegisterResponse(BaseModel): 
    message: str

# class TTSInlineRequest(BaseModel):
#     voice_id: str
#     text: str

class TTSPageFromListRequest(BaseModel):
    voice_id: str
    pages: List[str]
    page: int

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

class UserRequest(BaseModel):
    id: str
    passwd: str
    repasswd: str
    name: str
    address: str
      
class UserResponse(BaseModel):
    message: str

class LoginRequest(BaseModel):
    id: str
    passwd: str

class LoginResponse(BaseModel):
    uid: int
    name: str
    access_token: str
    token_type: str

class RecordCheckItem(BaseModel):
    fid: int
    type: int
    title: str
    summary: str
    contents: str
    create_date: date
    clips: int
    image_url: str | None

class RecordCheckResponse(BaseModel):
    uid: int
    records: List[RecordCheckItem]
    
class DetailResponse(BaseModel):
    uid: int
    type: int
    title: str
    summary: str
    contents: str
    create_dates: date
    image_url: str
    
class PageItem(BaseModel):
    text: Optional[str]
    image: Optional[str]

class FairyTaleItem(BaseModel):
    uid: int
    fid: int
    type: int
    title: str
    summary: str
    create_date: date
    pages: List[PageItem]

class SearchResponse(BaseModel):
    results: List[FairyTaleItem]

class UserUpdateRequest(BaseModel):
    uid: int
    id: str
    currentPasswd: str
    passwd: str
    repasswd: str
      
class UserUpdateResponse(BaseModel):
    message: str

class UserDeleteRequest(BaseModel):
    uid: int

class UserDeleteResponse(BaseModel):
    message: str

class UserUpdateSearchRequest(BaseModel):
    uid: int

class UserUpdateSearchResponse(BaseModel):
    id: str
    name: str
    address: str

class ReadRequest(BaseModel): 
    page: int
    voice_id: Optional[str] = None

class ResumeResponse(BaseModel):  
    uid: int
    fid: int
    total_pages: int
    last_page: int
    next_page: int


# 회원가입 API
@app.post("/join", response_model=UserResponse)
def join(req: UserRequest, db: Session = Depends(get_db)):
    if req.id == "":
        raise HTTPException(status_code=400, detail="아이디를 입력해주세요.")
    
    if req.passwd == "":
        raise HTTPException(status_code=400, detail="비밀번호를 입력해주세요.")
    
    if not bool(pattern.fullmatch(req.passwd)):
        raise HTTPException(status_code=400, detail="비밀번호에 대소문자, 특수문자, 숫자가 모두 입력됬는지 확인해주세요.")
    
    if req.repasswd == "":
        raise HTTPException(status_code=400, detail="비밀번호 재입력을 입력해주세요.")
    
    if req.passwd != req.repasswd:
        raise HTTPException(status_code=400, detail="비밀번호와 비밀번호 재입력이 일치하지 않습니다.")
    
    if req.name == "":
        raise HTTPException(status_code=400, detail="이름을 입력해주세요.")
    
    if req.address == "":
        raise HTTPException(status_code=400, detail="주소를 입력해주세요.")
    
    # 아이디 중복 체크
    existing = db.query(Users).filter(Users.id == req.id).first()
    if existing:
        raise HTTPException(status_code=400, detail="이미 존재하는 아이디입니다.")
    
    # 비밀번호 해싱
    hashed_passwd = pwd_context.hash(req.passwd)

    user = Users(
        id=req.id,
        passwd=hashed_passwd,
        name=req.name,
        address=req.address,
        useFlag=1,
        createDate=date.today(),
        updateDate=date.today(),
    )
      
    try:
        db.add(user)
        db.commit()
        db.refresh(user)
    except Exception as e:
      db.rollback()
      raise HTTPException(status_code=500, detail=f"서버 내부에 오류가 발생했습니다.")

    return UserResponse(message="회원가입이 완료되었습니다.")


# 비밀번호 체크
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


# JWT 토근 발행
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# 로그인 API
@app.post("/login", response_model=LoginResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    if req.id == "":
        raise HTTPException(status_code=400, detail="아이디를 입력해주세요.")
    
    if req.passwd == "":
        raise HTTPException(status_code=400, detail="비밀번호를 입력해주세요.")
    
    user = db.query(Users).filter(Users.id == req.id, Users.useFlag == 1).first()
    if not user or not verify_password(req.passwd, user.passwd):
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 잘못되었습니다.")
    
    access_token = create_access_token(
        data={"sub": user.uid},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return LoginResponse( 
        uid=user.uid,
        name=user.name,
        access_token=access_token, 
        token_type="bearer")

# 동화 생성 API
@app.post("/generate", response_model=GenerateStoryResponse)
def generate(req: GenerateStoryRequest, db: Session = Depends(get_db), _=Depends(verify_token)):
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

@app.post("/voices/register", response_model=VoiceRegisterResponse) 
async def register_voice(uid: int = Form(...), audio: UploadFile = File(...), db: Session = Depends(get_db), _=Depends(verify_token)):
    if not XI_API_KEY:
        raise HTTPException(status_code=500, detail="missing XI_API_KEY")

    try:
        # 디스크 저장 없이 스트리밍 업로드
        files = {"files": (audio.filename, await audio.read(), audio.content_type or "audio/mpeg")}
        data = {"name": f"{uid} voice", "description": "동화책 TTS 커스텀 목소리"}
        headers = {"xi-api-key": XI_API_KEY}

        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(ELEVEN_ADD_URL, headers=headers, data=data, files=files)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"voice_register_failed: {r.text}")

        voice_id = r.json().get("voice_id")
        if not voice_id:
            raise HTTPException(status_code=502, detail="voice_id_missing_from_provider")

        v = Voices(
            uid=uid,
            voice_id=voice_id,
            memo="", 
            voiceFile="",
            createDate=date.today()
        )
        db.add(v); db.flush(); db.refresh(v); db.commit()

        return VoiceRegisterResponse(message="사용자 음성 등록에 성공하였습니다.")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"register_internal_error: {e}")

@app.get("/users/{uid}/fairy_tales/{fid}/resume", response_model=ResumeResponse)
def resume_reading(uid: int, fid: int, db: Session = Depends(get_db)):
    result = reader.resume_reading(db, uid, fid)
    return ResumeResponse(**result)

@app.post("/users/{uid}/fairy_tales/{fid}/read")
def read_page(uid: int, fid: int, req: ReadRequest = Body(...), db: Session = Depends(get_db)):
    if not XI_API_KEY:
        raise HTTPException(status_code=500, detail="missing XI_API_KEY")
    return reader.stream_page(db, uid, fid, page=req.page, voice_id=req.voice_id)

@app.post("/tts/stream_page")
def tts_stream_page(req: TTSPageFromListRequest, db: Session = Depends(get_db)):
    if not XI_API_KEY:
        raise HTTPException(status_code=500, detail="missing XI_API_KEY")
    if not req.pages:
        raise HTTPException(status_code=400, detail="pages_required")
    if req.page < 1 or req.page > len(req.pages):
        raise HTTPException(status_code=400, detail=f"invalid_page_number: 1..{len(req.pages)}")

    text = (req.pages[req.page - 1] or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty_page_text")

    return StreamingResponse(
        sg.tts_generator(voice_id=req.voice_id, text=text),
        media_type="audio/mpeg",
        headers={"Content-Disposition": f'inline; filename="page{req.page}.mp3"'}
    )
    
# Backend API: 나의 독서기록 조회
@app.get("/users/{uid}/check_records", response_model=RecordCheckResponse)
def check_records(uid: int, db: Session = Depends(get_db)):
    rows = (
        db.query(FairyTale, FairyTaleLog, FairyTaleImages)
        .join(FairyTaleLog, FairyTale.fid == FairyTaleLog.fid)
        .outerjoin(FairyTaleImages, FairyTale.fid == FairyTaleImages.fid)
        .filter(FairyTaleLog.uid == uid)
        .group_by(FairyTaleLog.lid)
        .all()
    )

    if not rows:
        raise HTTPException(status_code=404, detail="읽은 기록이 없음")

    records = []
    for ft, log, img in rows:
        image_url = None
        if img and img.file_name:
            full_image_path = f"{img.image_path}/{img.file_name}"
            
            if os.path.exists(full_image_path):
                try:
                    with open(full_image_path, "rb") as image_file:
                        encoded = base64.b64encode(image_file.read()).decode()
                        if img.file_name.lower().endswith('.png'):
                            image_url = f"data:image/png;base64,{encoded}"
                        else:
                            image_url = f"data:image/jpeg;base64,{encoded}"
                except Exception as e:
                    print(f"Error encoding image: {e}")
                    image_url = None

        records.append(
            RecordCheckItem(
                fid=ft.fid,
                type=ft.type,
                title=ft.title,
                summary=ft.summary,
                contents=ft.contents,
                create_date=ft.createDate,
                clips=log.clip,
                image_url=image_url
            )
        )

    return RecordCheckResponse(uid=uid, records=records)

# 회원정보 수정 API
@app.post("/update_user", response_model=UserUpdateResponse)
def update_user(req: UserUpdateRequest, db: Session = Depends(get_db), _=Depends(verify_token)):
    if req.currentPasswd == "":
        raise HTTPException(status_code=400, detail="현재 비밀번호를 입력해주세요.")
    
    user = db.query(Users).filter(Users.id == req.id, Users.useFlag == 1).first()
    if not user or not verify_password(req.currentPasswd, user.passwd):
        raise HTTPException(status_code=401, detail="현재 비밀번호가 일치 하지 않습니다.")
    
    if req.passwd == "":
        raise HTTPException(status_code=400, detail="비밀번호를 입력해주세요.")
    
    if not bool(pattern.fullmatch(req.passwd)):
        raise HTTPException(status_code=400, detail="비밀번호에 대소문자, 특수문자, 숫자가 모두 입력됬는지 확인해주세요.")
    
    if req.repasswd == "":
        raise HTTPException(status_code=400, detail="비밀번호 재입력을 입력해주세요.")
    
    if req.passwd != req.repasswd:
        raise HTTPException(status_code=400, detail="비밀번호와 비밀번호 재입력이 일치하지 않습니다.")
    
    # 비밀번호 해싱
    hashed_passwd = pwd_context.hash(req.passwd)
    
    user.passwd = hashed_passwd
    user.updateDate = date.today()

    try:
        db.commit()
        db.refresh(user)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="서버 내부에 오류가 발생했습니다.")

    return UserUpdateResponse(message="회원정보 수정이 완료되었습니다.")
    
# Backend API: 동화책 상세정보 조회
@app.get("/users/{uid}/detail/{fid}", response_model=DetailResponse)
def book_detail(
    uid: int = Path(..., description="사용자 ID"),
    fid: int = Path(..., description="동화 ID"),
    db: Session = Depends(get_db)
):
    row = (
        db.query(FairyTale)
        .filter(FairyTale.fid == fid)
        .first()
    )

    if not row:
        raise HTTPException(status_code=404, detail="해당 동화를 찾을 수 없음")
    
    image_row = (
        db.query(FairyTaleImages)
        .filter(FairyTaleImages.fid == fid)
        .order_by(FairyTaleImages.image_id.asc())  # PK 기준 오름차순 → 첫 번째 이미지
        .first()
    )
    
    image_url = image_row.image_path if image_row else None

    return DetailResponse(
        uid=row.uid,
        type=row.type,
        title=row.title,
        summary=row.summary,
        contents=row.contents,
        create_dates=row.createDate,
        image_url=image_url
    )

# Backend API: 동화책 검색
@app.get("/users/{uid}/search", response_model=SearchResponse)
def search_books(
    uid: int = Path(..., description="사용자 ID"),
    fid: Optional[int] = Query(None, description="동화 ID"),
    type: Optional[int] = Query(None, description="기록 타입"),
    title: Optional[str] = Query(None, description="책 제목 (검색용)"),
    db: Session = Depends(get_db)):
    
    def _split_sentences_kor(text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s]

    def _split_into_chunks(contents: str) -> List[str]:
        sents = _split_sentences_kor(contents or "")
        chunks: List[str] = []
        for i in range(0, len(sents), 2):
            chunk = " ".join(sents[i:i+2]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def build_pages(chunks: List[str], images: List[str]) -> List[dict]:
        pages = []
        max_len = max(len(chunks), len(images))
        for i in range(max_len):
            text = chunks[i] if i < len(chunks) else None
            image = images[i] if i < len(images) else None
            pages.append({"text": text, "image": image})
        return pages

    query = db.query(FairyTale).filter(or_(FairyTale.uid == uid, FairyTale.uid == 0))

    # fid가 있으면 단일 검색
    if fid is not None:
        record = query.filter(FairyTale.fid == fid).first()
        if not record:
            raise HTTPException(status_code=404, detail="해당 동화를 찾을 수 없음")

        # contents 분리
        chunks = _split_into_chunks(record.contents)

        # 이미지 로드
        images = (
            db.query(FairyTaleImages)
            .filter(FairyTaleImages.fid == record.fid)
            .order_by(FairyTaleImages.image_id.asc())
            .all()
        )
        image_paths = [img.image_path for img in images]

        # 페이지 구성
        pages = build_pages(chunks, image_paths)

        return SearchResponse(
            results=[
                FairyTaleItem(
                    uid=record.uid,
                    fid=record.fid,
                    type=record.type,
                    title=record.title,
                    summary=record.summary,
                    create_date=record.createDate,
                    pages=[PageItem(**p) for p in pages]
                )
            ]
        )

    # type/title 필터 추가
    if type is not None:
        query = query.filter(FairyTale.type == type)
    if title:
        query = query.filter(FairyTale.title.contains(title))

    records = query.all()
    if not records:
        raise HTTPException(status_code=404, detail="검색 결과 없음")

    results = []
    for r in records:
        chunks = _split_into_chunks(r.contents)
        images = (
            db.query(FairyTaleImages)
            .filter(FairyTaleImages.fid == r.fid)
            .order_by(FairyTaleImages.image_id.asc())
            .all()
        )
        image_paths = [img.image_path for img in images]
        pages = build_pages(chunks, image_paths)

        results.append(
            FairyTaleItem(
                uid=r.uid,
                fid=r.fid,
                type=r.type,
                title=r.title,
                summary=r.summary,
                create_date=r.createDate,
                pages=[PageItem(**p) for p in pages]
            )
        )

    return SearchResponse(results=results)

# 회원 탈퇴 API
@app.post("/delete_user", response_model=UserDeleteResponse)
def delete_user(req: UserDeleteRequest,  db: Session = Depends(get_db), _=Depends(verify_token)):
    # 유저 조회
    user = db.query(Users).filter(Users.uid == req.uid, Users.useFlag == 1).first()
    if not user:
        raise HTTPException(status_code=404, detail="해당 사용자가 존재하지 않습니다.")

    # 탈퇴 처리 (soft delete)
    user.useFlag = 0
    user.updateDate = date.today()

    try:
        db.commit()
        db.refresh(user)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="서버 내부에 오류가 발생했습니다.")

    return UserDeleteResponse(message="회원탈퇴가 완료되었습니다.")

# 회원 정보수정 사용자 정보 조회
@app.post("/user_update_search", response_model=UserUpdateSearchResponse)
def user_search(req: UserUpdateSearchRequest, db: Session = Depends(get_db), _=Depends(verify_token)):
    user = db.query(Users).filter(Users.uid == req.uid, Users.useFlag == 1).first()
    if not user:
        raise HTTPException(status_code=404, detail="해당 사용자가 존재하지 않습니다.")

    return UserUpdateSearchResponse(
        id=user.id,
        name=user.name,
        address=user.address
    )

# 동화 목록 조회
@app.get("/api/fairy_tales/default")
def get_default_fairy_tales(db: Session = Depends(get_db)):
    """
    DB에 저장된 모든 동화 목록을 가져오는 API
    """
    fairy_tales_with_images = []
    
    # FairyTale 테이블에서 모든 동화를 조회
    all_tales = db.query(FairyTale).all()
    
    # 각 동화에 대해 이미지 정보 추가
    for tale in all_tales:
        # FairyTaleImages 테이블에서 해당 동화의 첫 번째 이미지 경로를 조회
        image = db.query(FairyTaleImages).filter(FairyTaleImages.fid == tale.fid).order_by(FairyTaleImages.image_id.asc()).first()
        image_data = None

        if image and os.path.exists(image.image_path):
            with open(image.image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # 동화 객체와 이미지 경로를 합쳐서 결과 리스트에 추가
        fairy_tales_with_images.append({
            "fid": tale.fid,
            "uid": tale.uid,
            "title": tale.title,
            "summary": tale.summary,
            "contents": tale.contents,
            "createDate": tale.createDate,
            "image": image_data, # base64로 인코딩된 이미지 데이터를 반환
        })

    return {"data": fairy_tales_with_images}

# 폴더 내 모든 이미지를 정렬된 순서로 조회
@app.get("/images/all")
async def get_all_images(folder_path: str):
    """폴더 내 모든 이미지를 정렬된 순서로 조회"""
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="폴더를 찾을 수 없습니다")
    
    # 이미지 파일 확장자
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if not image_files:
        raise HTTPException(status_code=404, detail="이미지 파일을 찾을 수 없습니다")
    
    image_files.sort()
    
    # 모든 이미지를 base64로 인코딩
    import base64
    images_data = []
    
    for i, image_file in enumerate(image_files):
        try:
            with open(image_file, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode()
                images_data.append({
                    "index": i + 1,
                    "filename": os.path.basename(image_file),
                    "image": image_data
                })
        except Exception as e:
            continue
    
    return {
        "folder_path": folder_path,
        "total_count": len(images_data),
        "images": images_data
    }