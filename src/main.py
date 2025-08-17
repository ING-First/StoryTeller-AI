from typing import Union, Optional
from fastapi import FastAPI, Depends, HTTPException, Query, Path
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from sqlalchemy.orm import Session
from db import SessionLocal
from db_models import Users, FairyTale, FairyTaleLog
from generate_summary import Summarizer
from datetime import date, datetime, timedelta
from passlib.context import CryptContext
from jose import jwt
from dotenv import load_dotenv
import os
import re

load_dotenv()
app = FastAPI()

REMOTE_GENERATE_URL = os.getenv("REMOTE_GENERATE_URL")
REQUEST_TIMEOUT = float(os.getenv("REMOTE_TIMEOUT", "5"))

summarizer = Summarizer()
summarizer.load_lora_model()

load_dotenv(override=False)

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

class UserRequest(BaseModel):
    id: str
    passwd: str
    repasswd: str
    name: str
    address: str
      
class UserResponse(BaseModel):
    message: str
    id: str

class LoginRequest(BaseModel):
    id: str
    passwd: str

class LoginResponse(BaseModel):
    message: str
    access_token: str
    token_type: str

class RecordCheckResponse(BaseModel):
    uid: int
    type: int
    title: str
    summary: str
    contents: str
    create_dates: date
    clips: int
    
class DetailResponse(BaseModel):
    uid: int
    type: int
    title: str
    summary: str
    contents: str
    create_dates: date
    
class SearchResponse(BaseModel):
    uid: int
    type: list[int]
    title: list[str]
    summary: list[str]
    contents: list[str]
    create_dates: list[date]

class UserUpdateRequest(BaseModel):
    id: str
    passwd: str
    repasswd: str
    name: str
    address: str
      
class UserUpdateResponse(BaseModel):
    message: str

class UserDeleteRequest(BaseModel):
    uid: int
    passwd: str

class UserDeleteResponse(BaseModel):
    message: str

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

    return {"message": "회원가입이 완료되었습니다.", "id": user.id}


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
        data={"sub": user.id},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return {"message": "로그인되었습니다.", "access_token": access_token, "token_type": "bearer"}


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
    
# Backend API: 나의 독서기록 조회
@app.get("/users/{uid}/check_records", response_model=RecordCheckResponse)
def check_records(
    uid: int, 
    fid: int = Query(..., description="동화 ID"),
    db: Session = Depends(get_db)
    ):
    
    row = (
        db.query(FairyTale, FairyTaleLog)
        .join(FairyTaleLog, FairyTale.fid == FairyTaleLog.fid)
        .filter(FairyTale.uid == uid, FairyTaleLog.uid == uid)
        .filter(FairyTale.fid == fid, FairyTaleLog.fid == fid)
        .first()
    )

    # 조회 기록이 없는 경우 Error Message 출력
    if not row:
        raise HTTPException(status_code=404, detail="기록을 찾을 수 없음")

    ft, log = row
    return RecordCheckResponse(
        uid=uid,
        type=ft.type,
        title=ft.title,
        summary=ft.summary,
        contents=ft.contents,
        create_dates=ft.createDate,
        clips=log.clip,
    )


# 회원정보 수정 API
@app.post("/update_user", response_model=UserUpdateResponse)
def join(req: UserUpdateRequest, db: Session = Depends(get_db)):
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
    
    # 기존 유저 조회
    user = db.query(Users).filter(Users.id == req.id, Users.useFlag == 1).first()
    if not user:
        raise HTTPException(status_code=404, detail="해당 사용자가 존재하지 않습니다.")
    
    # 비밀번호 해싱
    hashed_passwd = pwd_context.hash(req.passwd)
    
    user.passwd = hashed_passwd
    user.name = req.name
    user.address = req.address
    user.updateDate = date.today()

    try:
        db.commit()
        db.refresh(user)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="서버 내부에 오류가 발생했습니다.")

    return {"message": "회원정보 수정이 완료되었습니다."}
    
# Backend API: 동화책 상세정보 조회
@app.get("/users/{uid}/detail", response_model=DetailResponse)
def book_detail(
    uid: int,
    fid: int = Query(..., description="동화 ID"),
    db: Session = Depends(get_db)
):
    row = (
        db.query(FairyTale)
        .filter(FairyTale.uid == uid, FairyTale.fid == fid)
        .first()
    )

    if not row:
        raise HTTPException(status_code=404, detail="해당 동화를 찾을 수 없음")

    return DetailResponse(
        uid=row.uid,
        type=row.type,
        title=row.title,
        summary=row.summary,
        contents=row.contents,
        create_dates=row.createDate,
    )

# Backend API: 동화책 검색
@app.get("/users/{uid}/search", response_model=SearchResponse)
def search_books(
    uid: int = Path(..., description="사용자 ID"),
    type: Optional[int] = Query(None, description="기록 타입"),
    title: Optional[str] = Query(None, description="책 제목 (검색용)"),
    db: Session = Depends(get_db)):
    
    query = db.query(FairyTale)
    
    # uid 필터 검색
    if uid is not None:
        query = query.filter(FairyTale.uid == uid)
        
        # uid가 있는 경우에만 type 필터 검색
        if type is not None:
            query = query.filter(FairyTale.type == type)
    
    # 제목 필터 검색
    if title:
        query = query.filter(FairyTale.title.contains(title))
        
    records = query.all()
    
    return SearchResponse(
        uid=uid,
        type=[r.type for r in records],
        title=[r.title for r in records],
        summary=[r.summary for r in records],
        contents=[r.contents for r in records],
        create_dates=[r.createDate for r in records],
    )

# 회원 탈죄 API
@app.post("/delete_user", response_model=UserDeleteResponse)
def delete_user(req: UserDeleteRequest,  db: Session = Depends(get_db)):
    # 유저 조회
    user = db.query(Users).filter(Users.uid == req.uid, Users.useFlag == 1).first()
    if not user:
        raise HTTPException(status_code=404, detail="해당 사용자가 존재하지 않습니다.")

    # 비밀번호 확인
    if not pwd_context.verify(req.passwd, user.passwd):
        raise HTTPException(status_code=400, detail="비밀번호가 일치하지 않습니다.")

    # 탈퇴 처리 (soft delete)
    user.useFlag = 0
    user.updateDate = date.today()

    try:
        db.commit()
        db.refresh(user)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="서버 내부에 오류가 발생했습니다.")

    return {"message": "회원탈퇴가 완료되었습니다."}