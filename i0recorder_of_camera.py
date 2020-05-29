# https://stackoverflow.com/questions/59556761/how-to-start-and-stop-saving-video-frames-according-to-a-trigger-with-opencv-vid


from threading import Thread
import cv2

class RTSPVideoWriterObject(object):
    def __init__(self, src=0):
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(src)
        self.record = True

        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4))

        # Set up codec and output video settings
        self.codec = cv2.VideoWriter_fourcc(*"H264")
        self.output_video = cv2.VideoWriter('i0output.avi', self.codec, 30, (self.frame_width, self.frame_height))

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def show_frame(self):
        # Display frames in main program
        if self.status:
            cv2.imshow('frame', self.frame)
            if self.record:
                self.save_frame()

        # Press Q on keyboard to stop recording
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            self.output_video.release()
            cv2.destroyAllWindows()
            exit(1)
        # Press spacebar to start/stop recording
        elif key == 32:
            if self.record:
                self.record = False
                print('Stop recording')
            else:
                self.record = True
                print('Start recording')

    def save_frame(self):
        # Save obtained frame into video output file
        self.output_video.write(self.frame)

if __name__ == '__main__':
    rtsp_stream_link = 'rtsp://admin:admin123@10.248.10.100:554/cam/realmonitor?channel=1&subtype=0'
    video_stream_widget = RTSPVideoWriterObject(rtsp_stream_link)
    while True:
        try:
            video_stream_widget.show_frame()
        except AttributeError:
            pass
