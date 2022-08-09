from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from routers import face

app = FastAPI(
  title="Ignis Face Api",
  version='1.4.0'
)

origins = ["*"]

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

app.include_router(face.router)

@app.get("/")
def welcome():
  return {"message": "Welcome to Ignis Face API"}

if __name__ == '__main__':
  uvicorn.run(app, port=8000, host="0.0.0.0")