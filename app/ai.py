import cv2
from fastapi import HTTPException, status, UploadFile
from io import BytesIO
import mediapipe as mp
import math
import numpy as np
from numpy import asarray
from PIL import Image
import tensorflow as tf
from typing import  Tuple, Union

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

def read_imagefile(file) -> Image.Image:
  image = Image.open(BytesIO(file))
  return asarray(image)

def draw_rectangle(image, detection): 
  image_rows, image_cols, _ = image.shape
  location = detection.location_data

  def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
      return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                        math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
      # TODO: Draw coordinates even if it's outside of the image bounds.
      return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

  relative_bounding_box = location.relative_bounding_box
  rect_start_point = _normalized_to_pixel_coordinates(
    relative_bounding_box.xmin, 
    relative_bounding_box.ymin, 
    image_cols,
    image_rows
  )
  rect_end_point = _normalized_to_pixel_coordinates(
    relative_bounding_box.xmin + relative_bounding_box.width,
    relative_bounding_box.ymin + relative_bounding_box.height, 
    image_cols,
    image_rows
  )
  cv2.rectangle(image, rect_start_point, rect_end_point, (255, 0, 0), 2)
  return rect_start_point, rect_end_point

def found_face(image):
  extension = image.filename.split('.')[-1] in ('jpg', 'jpeg', 'png')
  if not extension:
    raise HTTPException(
      status_code=status.HTTP_404_NOT_FOUND, 
      detail=f"Invalid Image Format"
    )

  # Convert the uploaded image to np.ndarray
  ndarray_image = read_imagefile(image.file.read())

  with mp_face_mesh.FaceMesh(
    max_num_faces=10,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    results = face_mesh.process(ndarray_image)

    if not results.multi_face_landmarks:
      raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, 
        detail=f"Not found faces in the image"
      )
    
    number_faces = len(results.multi_face_landmarks)
    if number_faces > 1:
      raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST, 
        detail=f"Many faces found"
      )
    
  return True