from fastapi import APIRouter, HTTPException, status, UploadFile
from app import ai

router = APIRouter(
  prefix="/face",
  tags=['Face']
)

@router.post("/", status_code=status.HTTP_201_CREATED)
def found_face(image: UploadFile):
  return ai.found_face(image)

@router.post("/age", status_code=status.HTTP_201_CREATED)
def predict_age(image: UploadFile):
  try:
    age = ai.prediction_age(image)
    return age
  except:
    raise HTTPException(
      status_code=status.HTTP_404_NOT_FOUND, 
      detail=f"Not found face"
    )

@router.post("/gender", status_code=status.HTTP_201_CREATED)
def predict_gender(image: UploadFile):
  try:
    gender = ai.prediction_gender(image)
    return gender
  except:
    raise HTTPException(
      status_code=status.HTTP_404_NOT_FOUND, 
      detail=f"Not found face"
    )