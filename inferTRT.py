import numpy as np
import time
import cv2
from models import TrtModel

trtModule = TrtModel("yolov8s.engine", dtype = np.float16)

while True:

    img = cv2.imread("bus.jpg")

    start = time.time()
    trtModule(img)
    end = time.time()
    print(f"FPS: {1/(end-start)}")
