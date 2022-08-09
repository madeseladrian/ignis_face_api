from fastapi import APIRouter, HTTPException, status, UploadFile
from io import BytesIO
from PIL import Image
from numpy import asarray
from app import ai

router = APIRouter(
  prefix="/face",
  tags=['Face']
)

@router.post("/", status_code=status.HTTP_201_CREATED)
def found_face(image: UploadFile):
  return ai.found_face(image)