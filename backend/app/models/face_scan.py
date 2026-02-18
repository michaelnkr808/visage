from typing import Any
from datetime import datetime
from sqlalchemy import Column, Integer, String, LargeBinary, DateTime, ForeignKey, Float, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class Photo(Base):
    __tablename__ = "photos"
    
    id = Column[int](Integer, primary_key=True, autoincrement=True)
    filename = Column[str](String, nullable=True)
    image_data = Column[bytes](LargeBinary, nullable=False)
    created_at = Column[datetime](DateTime, default=func.now())
    
    transcript = relationship("Transcript", back_populates="photo", uselist=False, cascade="all, delete-orphan")  
    faces = relationship("DetectedFace", back_populates="photo", cascade="all, delete-orphan")


class Transcript(Base):
    __tablename__ = "transcripts"
    
    id = Column[int](Integer, primary_key=True, autoincrement=True)
    photo_id = Column[int](Integer, ForeignKey("photos.id", ondelete="CASCADE"), unique=True, nullable=False)
    raw_text = Column[str](Text, nullable=True)
    extracted_name = Column[str](String, nullable=True)
    context = Column[str](Text, nullable=True)
    created_at = Column[datetime](DateTime, default=func.now())
    
    photo = relationship("Photo", back_populates="transcript")


class DetectedFace(Base):
    __tablename__ = "detected_faces"
    
    id = Column[int](Integer, primary_key=True, autoincrement=True)
    photo_id = Column[int](Integer, ForeignKey("photos.id", ondelete="CASCADE"), nullable=False)  # ← ADDED ondelete and nullable
   
    # Bounding box coordinates
    x = Column[int](Integer, nullable=False)
    y = Column[int](Integer, nullable=False)
    width = Column[int](Integer, nullable=False)
    height = Column[int](Integer, nullable=False)
    
    # Cropped face
    face_image_data = Column[bytes](LargeBinary, nullable=True)
    confidence = Column[Any](Float, nullable=True)
    created_at = Column[datetime](DateTime, default=func.now())
    
    photo = relationship("Photo", back_populates="faces")
    encoding = relationship("FaceEncoding", back_populates="face", uselist=False, cascade="all, delete-orphan")
    person_info = relationship("PersonInfo", back_populates="face", uselist=False, cascade="all, delete-orphan")


class FaceEncoding(Base):
    __tablename__ = "face_encodings"
    
    id = Column[int](Integer, primary_key=True, autoincrement=True)
    face_id = Column[int](Integer, ForeignKey("detected_faces.id", ondelete="CASCADE"), unique=True, nullable=False)
    
    # 128 embedding vector
    encoding = Column[Any](Vector(128), nullable=False)
    
    model_name = Column[str](String, default="Facenet")
    created_at = Column[datetime](DateTime, default=func.now())
    
    face = relationship("DetectedFace", back_populates="encoding")


class PersonInfo(Base):
    __tablename__ = "person_info"
    
    id = Column[int](Integer, primary_key=True, autoincrement=True)
    face_id = Column[int](Integer, ForeignKey("detected_faces.id", ondelete="CASCADE"), unique=True, nullable=True)
    
    name = Column[str](String, nullable=True)
    conversation_context = Column[str](Text, nullable=True)
    
    first_met_at = Column[datetime](DateTime, default=func.now())
    last_seen_at = Column[datetime](DateTime, default=func.now())
    times_met = Column[int](Integer, default=1)
    
    face = relationship("DetectedFace", back_populates="person_info")