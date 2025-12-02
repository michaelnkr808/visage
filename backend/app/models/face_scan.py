from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime, ForeignKey, Float, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class Photo(Base):
    __tablename__ = "photos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=True)
    image_data = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, default=func.now())

    transcript = relationship("Transcript", back_populates="photo", uselist=False)
    faces = relationship("DetectedFace", back_populates="photo", cascade="all, delete-orphan")

class Transcript(Base):
    __tablename__ = "transcripts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    photo_id = Column(Integer, ForeignKey("photos.id", ondelete="CASCADE"), unique=True, nullable=False)
    raw_text = Column(Text, nullable=True)
    extracted_name = Column(String, nullable=True)
    context = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())

    photo = relationship("Photo", back_populates="transcript")

class DetectedFace(Base):
    __tablename__ = "detected_faces"

    id = Column(Integer, primary_key=True, autoincrement=True)
    photo_id = Column(Integer, ForeignKey("photos.id"))
   
    #bounding box coordinates
    x = Column(Integer, nullable=False)
    y = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)

    #cropped face
    face_image_data = Column(LargeBinary, nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=func.now())

    photo = relationship("Photo", back_populates="faces")
    encoding = relationship("FaceEncoding", back_populates="face", uselist=False, cascade="all, delete-orphan")
    person_info = relationship("PersonInfo", back_populates="face", uselist=False, cascade="all, delete-orphan")

class FaceEncoding(Base):
    __tablename__ = "face_encodings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    face_id = Column(Integer, ForeignKey("detected_faces.id", ondelete="CASCADE"), unique=True, nullable=False)

    #128 embedding vector
    encoding = Column(Vector(128), nullable=False)

    model_name = Column(String, default="Facenet")
    created_at = Column(DateTime, default=func.now())

    face = relationship("DetectedFace", back_populates="encoding")

class PersonInfo(Base):
    __tablename__ = "person_info"

    id = Column(Integer, primary_key=True, autoincrement=True)
    face_id = Column(Integer, ForeignKey("detected_faces.id", ondelete="CASCADE"), unique=True, nullable=False)

    name = Column(String, nullable=True)

    conversation_context = Column(Text, nullable=True)
    
    first_met_at = Column(DateTime, default=func.now())
    last_seen_at = Column(DateTime, default=func.now())
    times_met = Column(Integer, default=1)

    face = relationship("DetectedFace", back_populates="person_info", uselist=False)

