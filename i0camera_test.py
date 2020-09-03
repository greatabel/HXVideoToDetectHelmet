# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

# 119.100.21.43  8000 8001 8002  waiwang
# http://10.249.181.12 admin yxgl123456

# nvr 192.168.200.12
#
#
import cv2
cap = cv2.VideoCapture("rtsp://admin:huaxin12345@10.248.10.43:554/Streaming/Channels/1")
cap = cv2.VideoCapture("rtsp://admin:test1024@10.248.10.111/h264/ch33/main/av_stream")

# cap = cv2.VideoCapture("rtsp://admin:yxgl$666@192.168.200.215:554/Streaming/Channels/1")
# cap = cv2.VideoCapture("rtsp://admin:yxgl123456@192.168.200.150:554//Streaming/Channels/1")
# cap = cv2.VideoCapture("rtsp://admin:yxgl!123456@10.249.181.9:554/Streaming/Channels/101")
# cap = cv2.VideoCapture("rtsp://admin:yxgl!123456@10.249.181.9:554/Streaming/tracks/1701?starttime=20200602t063812z&endtime=20200602t064016z")
# cap = cv2.VideoCapture("rtsp://admin:hik12345+@10.248.204.200:554/Streaming/Channels/D2")
# cap = cv2.VideoCapture("rtsp://admin:hik12345+@10.248.204.200:554/h264/ch33/main/av_stream")
# cap = cv2.VideoCapture("rtsp://admin:hik12345+@10.248.204.200:554/Streaming/Channels/201")
import urllib
# # adr = urllib.parse.quote_plus("admin")+ ":" + urllib.parse.quote_plus("hik12345+")
# adr = urllib.parse.quote_plus("admin")+ ":" + urllib.parse.quote_plus("huaxin12345")
# print(adr)
# # cap = cv2.VideoCapture("rtsp://admin:hik12345%2B@10.248.204.201:554")
# cap = cv2.VideoCapture("rtsp://"+adr + "@10.248.10.43:554/Streaming/Channels/101")

# cap = cv2.VideoCapture("rtsp://admin:yxgl123456@192.168.200.150:554/Streaming/Channels/101?starttime=20200520t063812z&endtime=20200520t064816z")

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
    # print(frame, type(frame),'#', ret)
    if frame is not None:

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
