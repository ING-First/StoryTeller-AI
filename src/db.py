from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv(override=False)

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME")
DB_TZ   = os.getenv("DB_TIMEZONE", "Asia/Seoul")
DB_SSL_CA = os.getenv("DB_SSL_CA")

required_vars = {
    "DB_USER": DB_USER,
    "DB_PASSWORD": DB_PASSWORD,
    "DB_HOST": DB_HOST,
    "DB_NAME": DB_NAME,
}
missing = [k for k, v in required_vars.items() if not v]

if missing:
    raise RuntimeError(f"DB 환경변수 누락: {', '.join(missing)}")

QUERY_OPTS = (
    f"?charset=utf8mb4"
    f"&autocommit=false"
    f"&local_infile=0"
    f"&connect_timeout=10"
    f"&init_command=SET time_zone='%2B09:00'"
)

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}{QUERY_OPTS}"

# 커넥션 풀 설정 추가
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=30,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)