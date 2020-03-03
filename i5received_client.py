# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

import cv2
''' 
旧方法有问题：Expected boundary '--' not found 的问题

'''

# cap = cv2.VideoCapture("http://0.0.0.0:5000/video_feed0")

# count = 0

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     print('#'*10, ret, frame, '#'*5)
#     if ret:
#     # Our operations on the frame come here
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Display the resulting frame
#         # cv2.imshow('frame',gray)
#         cv2.imshow('frame',frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # count += 38
#     # cap.set(1, count)

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

import numpy as np
import urllib.request

stream = urllib.request.urlopen('http://0.0.0.0:5000/video_feed0')
total_bytes = b''
while True:
    total_bytes += stream.read(1024)
    b = total_bytes.find(b'\xff\xd9') # JPEG end
    if not b == -1:
        a = total_bytes.find(b'\xff\xd8') # JPEG start
        jpg = total_bytes[a:b+2] # actual image
        total_bytes= total_bytes[b+2:] # other informations
        
        # decode to colored image ( another option is cv2.IMREAD_GRAYSCALE )
        img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR) 
        # cv2.imshow('Window name',cv2.flip(img, -1)) # display image while receiving data
        cv2.imshow('frame',img)
        if cv2.waitKey(1) ==27: # if user hit esc            
            break
cv2.destroyWindow()
