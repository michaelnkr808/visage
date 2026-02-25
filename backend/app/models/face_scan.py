from typing import Optional
from datetime import datetime
from sqlalchemy import String, LargeBinary, DateTime, ForeignKey, Float, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

class Base(DeclarativeBase):
    pass

class Photo(Base):
    __tablename__ = "photos"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    filename: Mapped[Optional[str]] = mapped_column(String)
    image_data: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now)
    
    transcript: Mapped[Optional["Transcript"]] = relationship(back_populates="photo", uselist=False, cascade="all, delete-orphan")
    faces: Mapped[list["DetectedFace"]] = relationship(back_populates="photo", cascade="all, delete-orphan")


class Transcript(Base):
    __tablename__ = "transcripts"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    photo_id: Mapped[int] = mapped_column(ForeignKey("photos.id", ondelete="CASCADE"), unique=True)
    raw_text: Mapped[Optional[str]] = mapped_column(Text)
    extracted_name: Mapped[Optional[str]] = mapped_column(String)
    context: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now)
    
    photo: Mapped["Photo"] = relationship(back_populates="transcript")


class DetectedFace(Base):
    __tablename__ = "detected_faces"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    photo_id: Mapped[int] = mapped_column(ForeignKey("photos.id", ondelete="CASCADE"))
   
    # Bounding box coordinates
    x: Mapped[int] = mapped_column()
    y: Mapped[int] = mapped_column()
    width: Mapped[int] = mapped_column()
    height: Mapped[int] = mapped_column()
    
    # Cropped face
    face_image_data: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now)
    
    photo: Mapped["Photo"] = relationship(back_populates="faces")
    encoding: Mapped[Optional["FaceEncoding"]] = relationship(back_populates="face", uselist=False, cascade="all, delete-orphan")
    person_info: Mapped[Optional["PersonInfo"]] = relationship(back_populates="face", uselist=False, cascade="all, delete-orphan")


class FaceEncoding(Base):
    __tablename__ = "face_encodings"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    face_id: Mapped[int] = mapped_column(ForeignKey("detected_faces.id", ondelete="CASCADE"), unique=True)
    
    # 128 embedding vector
    encoding: Mapped[Vector] = mapped_column(Vector(128))
    
    model_name: Mapped[str] = mapped_column(String, default="Facenet")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now)
    
    face: Mapped["DetectedFace"] = relationship(back_populates="encoding")


class PersonInfo(Base):
    __tablename__ = "person_info"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    face_id: Mapped[Optional[int]] = mapped_column(ForeignKey("detected_faces.id", ondelete="CASCADE"), unique=True)
    
    name: Mapped[Optional[str]] = mapped_column(String)
    conversation_context: Mapped[Optional[str]] = mapped_column(Text)
    
    first_met_at: Mapped[datetime] = mapped_column(DateTime, default=func.now)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=func.now)
    times_met: Mapped[int] = mapped_column(default=1)
    
    face: Mapped[Optional["DetectedFace"]] = relationship(back_populates="person_info")
