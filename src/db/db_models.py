from sqlalchemy import (
    Column, Integer, String, Text, Date, ForeignKey, SmallInteger
)
from sqlalchemy.orm import declarative_base, relationship
from datetime import date

Base = declarative_base()

# Users 테이블 정의
class Users(Base):
    __tablename__ = "Users"
    
    uid = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    id = Column(String(10), nullable=False)
    passwd = Column(String(256), nullable=False)
    name = Column(String(10), nullable=False)
    address = Column(String(256), nullable=False)
    useFlag = Column(SmallInteger, nullable=False)
    createDate = Column(Date, nullable=False)
    updateDate = Column(Date, nullable=False)
    
    fairy_tales = relationship("FairyTale", back_populates="users")
    logs = relationship("FairyTaleLog", back_populates="users")
    voices = relationship("Voices", back_populates="user")


# FairyTale 테이블 정의
class FairyTale(Base):
    __tablename__ = "FairyTale"

    fid = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    uid = Column(Integer, ForeignKey("Users.uid"), nullable=False)
    type = Column(SmallInteger, nullable=False)
    title = Column(String(50), nullable=False)
    summary = Column(String(256), nullable=False)
    contents = Column(Text)
    createDate = Column(Date, nullable=False)
    
    users = relationship("Users", back_populates="fairy_tales")
    logs = relationship("FairyTaleLog", back_populates="fairy_tale")
    images = relationship("FairyTaleImages", back_populates="fairy_tale")


# FairyTaleImage 테이블 정의
class FairyTaleImages(Base):
    __tablename__ = "FairyTaleImages"

    image_id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    fid = Column(Integer, ForeignKey("FairyTale.fid"), nullable=False)
    image_path = Column(String(256), nullable=False)
    file_name = Column(String(50), nullable=False)
    createDate = Column(Date, nullable=False)
    
    fairy_tale = relationship("FairyTale", back_populates="images")


# FairyTaleLog 테이블 정의
class FairyTaleLog(Base):
    __tablename__ = "FairyTaleLog"

    lid = Column(Integer, primary_key=True, autoincrement=True, nullable=False, comment="로그 pkey")
    fid = Column(Integer, ForeignKey("FairyTale.fid"), nullable=False, comment="동화 fid")
    uid = Column(Integer, ForeignKey("Users.uid"), nullable=False, comment="유저 id")
    clip = Column(Integer, nullable=False, comment="책갈피 페이지")
    createDate = Column(Date, nullable=False, comment="생성일자")
    updateDate = Column(Date, nullable=False, comment="수정일자")

    # 관계 설정
    users = relationship("Users", back_populates="logs")
    fairy_tale = relationship("FairyTale", back_populates="logs")


class Voices(Base): 
    __tablename__ = "Voices"
    vid = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    uid = Column(Integer, ForeignKey("Users.uid"), nullable=False)
    voice_id = Column(String(50), nullable=False)     
    memo = Column(String(10), nullable=False)         
    voiceFile = Column(String(100), nullable=False)   
    createDate = Column(Date, nullable=False, default=date.today)
    
    user = relationship("Users", back_populates="voices")