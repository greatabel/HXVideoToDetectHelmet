import cv2 as cv
import numpy as np


class video_analyse:
    def __init__(self):
        self.__video_capture = cv.VideoCapture('video/test.mp4')
        self.__fps = self.__video_capture.get(cv.CAP_PROP_FPS)
        self.__size = (int(self.__video_capture.get(cv.CAP_PROP_FRAME_WIDTH)),
                       int(self.__video_capture.get(cv.CAP_PROP_FRAME_HEIGHT)))
        print(self.__size)
        self.__f_nums = self.__video_capture.get(cv.CAP_PROP_FRAME_COUNT)
        with open('zeros.csv', 'w') as file_init:
            file_init.close()

    def show_analyse(self):
        a = 200
        b = 925
        c = 1075
        d = 1775
        success, frame = self.__video_capture.read()
        frame_gray_old = cv.cvtColor(frame[a: b, c: d], cv.COLOR_RGB2GRAY)

        image_range = frame_gray_old.copy()
        for i in range(0, image_range.shape[0]):
            for j in range(0, image_range.shape[1]):
                image_range[i, j] = 0
        pts = np.array([[165, 20], [10, 700], [680, 700], [385, 20]], np.int32)
        image_range = cv.fillConvexPoly(image_range, pts, 255)
        '''
        # for test
        image_zero = image_range.copy()
        for i in range(0, image_zero.shape[0]):
            for j in range(0, image_zero.shape[1]):
                image_zero[i, j] = 0    
        '''
        count = 0
        while success:
            if count % 25 == 0 and count != 0:
                frame_gray = cv.cvtColor(frame[a: b, c: d], cv.COLOR_RGB2GRAY)
                with open('zeros.csv', 'a') as f:
                    f.write(str(self.__sobel_image(frame_gray, image_range)) + "\n")
                print("frame %d is OK!" % count)
                self.__diff_image(frame_gray_old, frame_gray)
                frame_gray_old = frame_gray
                cv.waitKey(0)
            count += 1
            success, frame = self.__video_capture.read()

    def __sobel_image(self, image, image_range):
        sobelx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
        sobely = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
        gm = cv.sqrt(sobelx ** 2 + sobely ** 2)
        count_zero = 0
        for i in range(0, gm.shape[0]):
            for j in range(0, gm.shape[1]):
                if gm[i, j] == 0.0 and image_range[i, j] == 255:
                    count_zero += 1

        # for test
        cv.imshow('image', image)
        print(count_zero)
        self.__show_test_image(gm, image_range, image_range)

        return count_zero

    def __diff_image(self, image1, image2):
        # cv.imshow("diff", cv.absdiff(image1, image2))
        return

    def __show_test_image(self, image_1, image_2, image_3):
        merged_img = np.dstack([image_1, image_2, image_3])
        cv.imshow("outpic", merged_img)
        cv.waitKey(10)
