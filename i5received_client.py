# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

import cv2
cap = cv2.VideoCapture("http://0.0.0.0:5000/video_feed0")

count = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    print('#'*10, ret, frame, '#'*5)
    if ret:
    # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        # cv2.imshow('frame',gray)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # count += 38
    # cap.set(1, count)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
