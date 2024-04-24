import cv2
import numpy as np
import time
import PoseModule as pm
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json 


app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def check():
    return {"message": "Hello World!"}

@app.post("/process_video/")
async def process_video(video: UploadFile = File(...)):
    try:
        with open('video.mp4', 'wb') as handler:
            handler.write(video.file.read())
        cap = cv2.VideoCapture("video.mp4")
        if not cap.isOpened():
            print("Error opening video file")
            exit()

        detector = pm.poseDetector()
        count = 0
        dir = 0
        pTime = 0

        while True:
            success, img = cap.read()
            if not success:
                break # Break the loop if the video ends

            img = cv2.resize(img, (1280, 720))
            img = detector.findPose(img, False)
            lmList = detector.findPosition(img, False)

            if len(lmList) != 0:
                angle = detector.findAngle(img, 12, 14, 16)
                per = np.interp(angle, (210, 310), (0, 100))
                bar = np.interp(angle, (220, 310), (650, 100))

                color = (255, 0, 255)
                if per == 100:
                    color = (0, 255, 0)
                    if dir == 0:
                        count += 0.5
                        dir = 1
                if per == 0:
                    color = (0, 255, 0)
                    if dir == 1:
                        count += 0.5
                        dir = 0

                cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
                cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
                cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

                cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
            cv2.imshow("Image", img)
            cv2.waitKey(20) # Adjusted for smoother display

            if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
                break

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
   uvicorn.run("main:app", port=8000, log_level="info", reload=True)