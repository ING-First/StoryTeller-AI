from sqlalchemy import (
    Column, Integer, String, Text, Date, ForeignKey, SmallInteger
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

# Users
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