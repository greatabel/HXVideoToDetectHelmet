import cv2


if __name__ == '__main__' :

    # video = cv2.VideoCapture("test1.mp4")
    video = cv2.VideoCapture("rtsp://admin:admin123@10.248.10.133:554/cam/realmonitor?channel=3&subtype=1")

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    video.release()

