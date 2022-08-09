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

def resize_face(image: UploadFile):
  with mp_face_detection.FaceDetection(
      min_detection_confidence=0.5) as face_detection:
    
    # Convert the uploaded image to np.ndarray
    image = read_imagefile(image.file.read())   
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for detection in results.detections:
      rect_start_point, rect_end_point = draw_rectangle(image, detection)
      x1 = rect_start_point[0]
      y1 = rect_start_point[1]
      x2 = rect_end_point[0]
      y2 = rect_end_point[1]
      face = image[y1:y2, x1:x2, :]
      face = cv2.resize(face, (160, 160), interpolation=cv2.INTER_AREA)
      
  return face

def resize_face_shape(image: UploadFile):
  with mp_face_detection.FaceDetection(
      min_detection_confidence=0.5) as face_detection:
    
    # Convert the uploaded image to np.ndarray
    image = read_imagefile(image.file.read())   
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for detection in results.detections:
      margin = 0.15
      rect_start_point, rect_end_point = draw_rectangle(image, detection)
      correction = int(abs(rect_end_point[0] - rect_start_point[0]) * margin) 
      x1 = rect_start_point[0] - correction
      y1 = rect_start_point[1] - correction
      x2 = rect_end_point[0] + correction
      y2 = rect_end_point[1] + correction
      face = image[y1:y2, x1:x2, :]
      face = cv2.resize(face, (160, 160), interpolation=cv2.INTER_AREA)
      
  return face

def prediction_age(image):
  model = tf.keras.models.load_model('models/age')
  class_names = [
    '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', 
    '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', 
    '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', 
    '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', 
    '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', 
    '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', 
    '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', 
    '8', '80', '9'
  ]

  image_resized = resize_face(image)
  # img_array = tf.keras.utils.img_to_array(image_resized)
  img_array = tf.expand_dims(image_resized, 0) # Create a batch

  predictions = model.predict(img_array)
  predictions = tf.nn.softmax(predictions[0])

  prediction_age = 0
  for i in range(len(predictions)):
    prediction_age += predictions[i] * int(class_names[i])

  prediction_age = int(round(prediction_age.numpy()))
  return prediction_age

def prediction_gender(image):
  model = tf.keras.models.load_model('models/gender')
  class_names = ['Female', 'Male']

  image_resized = resize_face(image)
  # img_array = tf.keras.utils.img_to_array(image)
  img_array = tf.expand_dims(image_resized, 0)

  predictions = model.predict(img_array)
  predictions = tf.nn.sigmoid(predictions[0])
  score = tf.where(predictions < 0.5, 0, 1)

  return class_names[int(score)]

def prediction_face_shape(image):
  model = tf.keras.models.load_model('models/face_shape')
  class_names = ['heart', 'oblong', 'oval', 'round', 'square']

  image_resized = resize_face_shape(image)
  # img_array = tf.keras.utils.img_to_array(image)
  img_array = tf.expand_dims(image_resized, 0)

  predictions = model.predict(img_array)
  predictions = tf.nn.softmax(predictions[0])

  return class_names[np.argmax(predictions)]
