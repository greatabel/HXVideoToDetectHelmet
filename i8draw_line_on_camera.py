# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

import cv2
cap = cv2.VideoCapture("rtsp://admin:admin123@10.248.10.100:554/cam/realmonitor?channel=3&subtype=0")
#cap = cv2.VideoCapture("test2.dav")
#cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"H264"))
# cap.set(3, 1920)
# cap.set(4, 1080)
# cap.set(cv2.CAP_PROP_FPS,5)
#cap = cv2.VideoCapture(0)

count = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else:
        cv2.line(img=frame, pt1=(10, 10), pt2=(100, 10), color=(255, 0, 0), thickness=5, lineType=8, shift=0)

    # Display the resulting frame
    # cv2.imshow('frame',gray)
    cv2.imshow('frame',frame)
    
    # count += 38
    # cap.set(1, count)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
