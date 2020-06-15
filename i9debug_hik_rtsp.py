import cv2
# 海康
#rtsp://admin:hik12345+@10.248.204.200:554
#rtsp://admin:yxgl!123456@10.249.181.9:554/Streaming/Channels/101
# 大华
input_str = 'rtsp://admin:admin123@10.248.10.100:554/cam/realmonitor?channel=1&subtype=0'
#rtsp://admin:admin123@10.248.10.100:554/cam/realmonitor?channel=3&subtype=0
# input_str = 'rtsp://admin:yxgl!123456@10.249.181.9:554/Streaming/Channels/101'

# input_str = 'rtsp://admin:huaxin12345@10.248.10.43:554:554/Streaming/Channels/101'

# input_str = 'rtsp://admin:huaxin12345@10.248.10.43:554:554/Streaming/Channels/102'
import urllib



def deal_specialchar_in_url(istr):
	s = istr.find('//')
	e = istr.find('@')
	# print(s, e)
	subs = istr[s+2: e]
	name, ps = subs.split(':')
	encoded_name_ps = urllib.parse.quote_plus(name)+ ":" + urllib.parse.quote_plus(ps)
	result = 'rtsp://' + encoded_name_ps + istr[e:]
	# print(result)
	return result

addr = deal_specialchar_in_url(input_str)
print('adr', addr)

# adr = urllib.parse.quote_plus("admin")+ ":" + urllib.parse.quote_plus("hik12345+")
# print(adr)
# cap = cv2.VideoCapture("rtsp://admin:hik12345%2B@10.248.204.201:554")
# cap = cv2.VideoCapture("rtsp://"+adr + "@10.248.204.201:554")
cap = cv2.VideoCapture(addr)
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


cap.release()
cv2.destroyAllWindows()