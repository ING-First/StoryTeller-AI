from typing import Optional, List
from fastapi.security import OAuth2PasswordBearer, HTTPBearer
from fastapi import FastAPI, Depends, HTTPException, UploadFile,  Query, Path, File, Form, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import or_
from sqlalchemy.orm import Session
from db.db_connector import SessionLocal
from db.db_models import Users, FairyTale, FairyTaleLog, Voices, FairyTaleImages
from datetime import date, datetime, timedelta
from generate_story.story_reading import StoryReader
from passlib.context import CryptContext
from jose import jwt, JWTError
from dotenv import load_dotenv
import uuid

from generate_story.generate_sound import SoundGenerator
import os
import re
import httpx
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

pattern = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*()_\-+=\[\]{}\\|;:\'",.<>/?`~])[A-Za-z\d!@#$%^&*()_\-+=\[\]{}\\|;:\'",.<>/?`~]{8,15}$')

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(
        token: str = Depends(oauth2_scheme),
        db : Session = Depends(get_db)
    ) -> Users:

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        print("Decoded payload:", payload) 
        uid: int = int(payload.get("sub"))
        if uid is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except JWTError:
        raise HTTPException(status_code=401, detail="=Token decode error")
    
    user = db.query(Users),filter(Users.uid == uid).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

class VoiceRegisterResponse(BaseModel): 
    message: str
    voice_id: str

class TTSPageFromListRequest(BaseModel):
    voice_id: str
    pages: List[str]
    page: int

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

class ReadRequest(BaseModel): 
    page: int
    voice_id: Optional[str] = None

class ResumeResponse(BaseModel):  
    uid: int
    fid: int
    total_pages: int
    last_page: int
    next_page: int

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
    name: str
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
class UpdateReadingProgressRequest(BaseModel):
    page: int

class UpdateReadingProgressResponse(BaseModel):
    message: str
    page: int

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


# JWT 토큰 발행
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

@app.post("/voices/register", response_model=VoiceRegisterResponse) 
async def register_voice(uid: int = Form(...), audio: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        save_dir = "ref_voices"
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"user_{uid}_{uuid.uuid4().hex[:8]}.wav")

        with open(file_path, "wb") as f:
            f.write(await audio.read())

        voice_id = f"voice_{uid}_{uuid.uuid4().hex[:8]}"

        v = Voices(
            uid=uid,
            voice_id=voice_id, 
            memo="",
            voiceFile=file_path,
            createDate=date.today()
        )
        db.add(v)
        db.commit()
        db.refresh(v)

        return {"message": "사용자 음성 등록 성공", "voice_id": voice_id} 
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"register_internal_error: {e}")

@app.get("/users/{uid}/fairy_tales/{fid}/resume", response_model=ResumeResponse)
def resume_reading(uid: int, fid: int, db: Session = Depends(get_db)):
    result = reader.resume_reading(db, uid, fid)
    return ResumeResponse(**result)

@app.post("/users/{uid}/fairy_tales/{fid}/read")
def read_page(uid: int, fid: int, req: ReadRequest = Body(...), db: Session = Depends(get_db)):
    v = (
        db.query(Voices)
        .filter(Voices.uid == uid)
        .order_by(Voices.vid.desc())
        .first()
    )
    voice_id = req.voice_id or getattr(v, "voice_id", None) 

    if not voice_id:
        raise HTTPException(status_code=400, detail="등록된 음성이 없습니다.")

    return reader.stream_page(db, uid, fid, page=req.page, voice_id=voice_id)  # 수정됨

@app.post("/tts/stream_page")
def tts_stream_page(uid: int = Body(...), pages: list[str] = Body(...), page: int = Body(...), db: Session = Depends(get_db)):
    if not pages:
        raise HTTPException(status_code=400, detail="pages_required")
    if page < 1 or page > len(pages):
        raise HTTPException(status_code=400, detail=f"invalid_page_number: 1..{len(pages)}")

    text = (pages[page - 1] or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty_page_text")

    v = (
        db.query(Voices)
        .filter(Voices.uid == uid)
        .order_by(Voices.vid.desc())
        .first()
    )
    voice_id = getattr(v, "voice_id", None) if v else None  
    if not voice_id:
        raise HTTPException(status_code=400, detail="등록된 음성이 없습니다.")

    return StreamingResponse(
        sg.tts_generator(voice_id=voice_id, text=text), 
        media_type="audio/wav",
        headers={"Content-Disposition": f'inline; filename="page{page}.wav"'}
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
def update_user(req: UserUpdateRequest, db: Session = Depends(get_db)):
    if req.currentPasswd == "":
        raise HTTPException(status_code=400, detail="현재 비밀번호를 입력해주세요.")
    
    user = db.query(Users).filter(Users.id == req.id, Users.useFlag == 1).first()
    if not user or not verify_password(req.currentPasswd, user.passwd):
        raise HTTPException(status_code=401, detail="현재 비밀번호가 일치 하지 않습니다.")
    
    if req.passwd != "":
        if  not bool(pattern.fullmatch(req.passwd)):
            raise HTTPException(status_code=400, detail="비밀번호에 대소문자, 특수문자, 숫자가 모두 입력됬는지 확인해주세요.")
        
        if req.repasswd == "":
            raise HTTPException(status_code=400, detail="비밀번호 재입력을 입력해주세요.")
        
        if req.passwd != req.repasswd:
            raise HTTPException(status_code=400, detail="비밀번호와 비밀번호 재입력이 일치하지 않습니다.")
    
    user.name = req.name
    
    if req.passwd != "":
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
        
        log = db.query(FairyTaleLog).filter(FairyTaleLog.fid == fid, FairyTaleLog.uid == uid).first()
        print(log)
        if not log:
            f = FairyTaleLog(
                uid=uid,
                fid=fid,
                clip=1, 
                createDate=date.today(),
                updateDate=date.today()
            )
            db.add(f); db.flush(); db.refresh(f); db.commit()

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
def delete_user(req: UserDeleteRequest,  db: Session = Depends(get_db)):
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
def user_search(req: UserUpdateSearchRequest, db: Session = Depends(get_db)):
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
    DB에 저장된 모든 동화 목록을 가져오는 API (중복 제거)
    """
    fairy_tales_with_images = []

    # FairyTale과 FairyTaleImages를 조인해서 각 동화별 첫 번째 이미지만 가져오기
    tales_with_images = (
        db.query(
            FairyTale,
            FairyTaleImages
        )
        .outerjoin(FairyTaleImages, FairyTale.fid == FairyTaleImages.fid)
        .filter(FairyTale.uid == 0)  # uid가 0인 동화만 (기본 동화)
        
        .group_by(FairyTale.fid)  # fid 기준으로 그룹화 (중복 제거)
        .order_by(FairyTale.fid, FairyTaleImages.image_id.asc())
        .all()
    )

    for tale, image in tales_with_images:
        image_data = None
        if image and image.file_name:
            full_image_path = f"{image.image_path}/{image.file_name}"
            if os.path.exists(full_image_path):
                try:
                    with open(full_image_path, "rb") as image_file:
                        encoded = base64.b64encode(image_file.read()).decode()
                        if image.file_name.lower().endswith('.png'):
                            image_data = f"data:image/png;base64,{encoded}"
                        else:
                            image_data = f"data:image/jpeg;base64,{encoded}"
                except Exception as e:
                    print(f"Error encoding image: {e}")
                    image_data = None

        fairy_tales_with_images.append({
            "fid": tale.fid,
            "uid": tale.uid,
            "title": tale.title,
            "summary": tale.summary,
            "contents": tale.contents,
            "createDate": tale.createDate,
            "image": image_data,  # base64 인코딩된 이미지
        })

    return {"data": fairy_tales_with_images}

# 로그인 사용자용 동화 목록 조회
@app.get("/api/fairy_tales/my")
def get_my_fairy_tales( 
    db : Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """
        로그인한 사용자의 동화 목록만 가져오기 
    """
    fairy_tales_with_images = []

    tales_with_images = (
        db.query(FairyTale, FairyTaleImages)
        .outerjoin(FairyTaleImages, FairyTale.fid)
        .filter(FairyTale.uid == current_user.uid)   # <--- 로그인한 사용자 동화만 필터링

        .group_by(FairyTale.fid)
        .order_by(FairyTale.fid , FairyTaleImages.image_id.asc())
        .all()
    )

    for tale, image in tales_with_images:
        image_data = None
        if image and image.file_name:
            full_image_path = f"{image.image_path}/{image.file_name}"
            if os.path.exists(full_image_path):
                try:
                    with open(full_image_path, "rb") as image_file:
                        encoded = base64.b64encode(image_file.read()).decode()
                        if image.file_name.lower().endswith('.png'):
                            image_data = f"data:image/png;base64,{encoded}"
                        else:
                            image_data = f"data:image/jpeg;base64,{encoded}"
                except Exception as e:
                    print(f"Error encoding image: {e}")
                    image_data = None

        fairy_tales_with_images.append({
            "fid": tale.fid,
            "uid": tale.uid,
            "title": tale.title,
            "summary": tale.summary,
            "contents": tale.contents,
            "createDate": tale.createDate,
            "image": image_data,
        })

    return {"data": fairy_tales_with_images}

# 로그인 사용자용 동화 목록 조회
@app.get("/api/fairy_tales/my")
def get_my_fairy_tales( 
    db : Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """
        로그인한 사용자의 동화 목록만 가져오기 
    """
    fairy_tales_with_images = []

    tales_with_images = (
        db.query(FairyTale, FairyTaleImages)
        .outerjoin(FairyTaleImages, FairyTale.fid)
        .filter(FairyTale.uid == current_user.uid)   # <--- 로그인한 사용자 동화만 필터링

        .group_by(FairyTale.fid)
        .order_by(FairyTale.fid , FairyTaleImages.image_id.asc())
        .all()
    )

    for tale, image in tales_with_images:
        image_data = None
        if image and image.file_name:
            full_image_path = f"{image.image_path}/{image.file_name}"
            if os.path.exists(full_image_path):
                try:
                    with open(full_image_path, "rb") as image_file:
                        encoded = base64.b64encode(image_file.read()).decode()
                        if image.file_name.lower().endswith('.png'):
                            image_data = f"data:image/png;base64,{encoded}"
                        else:
                            image_data = f"data:image/jpeg;base64,{encoded}"
                except Exception as e:
                    print(f"Error encoding image: {e}")
                    image_data = None

        fairy_tales_with_images.append({
            "fid": tale.fid,
            "uid": tale.uid,
            "title": tale.title,
            "summary": tale.summary,
            "contents": tale.contents,
            "createDate": tale.createDate,
            "image": image_data,
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