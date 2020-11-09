import cv2
import numpy as np

# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
width = 512
height = 910
video=cv2.VideoWriter('video.avi', fourcc, 15,(height, width))

for i in range(1,480):
 img = cv2.imread('screenshots/'+ str(i)+'.jpg')
 video.write(img)

cv2.destroyAllWindows()
video.release()