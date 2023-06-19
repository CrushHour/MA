import numpy as np
import cv2 as cv
from datetime import datetime

if __name__ == '__main__':

    video_path = '/home/robotlab/Documents/GitHub/MA_Schote/MA/Data/Video'

    start_time = datetime.now()
    dt = 0 
    cap = cv.VideoCapture(0)
    video_list = []

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while dt <= 100:
        t = datetime.now()
        dt = t - start_time
        dt = dt.seconds
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Display the resulting frame
        cv.imshow('frame', frame)
        video_list.append(frame)

        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv.VideoWriter(f'{video_path}/{start_time}.avi',cv.VideoWriter_fourcc(*'DIVX'), frameSize=(frame_width,frame_height), fps=25)
    for i in range(len(video_list)):
        frame = video_list[i]
        out.write(frame)
    # release camera
    cap.release()
    cv.destroyAllWindows()