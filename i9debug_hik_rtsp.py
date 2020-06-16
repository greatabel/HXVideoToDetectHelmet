import cv2
# 海康
# 12345
# yxgl$666
# yxgl123456
# input_str= 'rtsp://admin:yxgl$666@192.168.200.182:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.183:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.184:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.185:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.186:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.187:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.188:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.189:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.190:554'

# camera is error 
# input_str= 'rtsp://admin:yxgl$666@192.168.200.191:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.192:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.193:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.194:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.197:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.198:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.202:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.205:554'

# camear is error(error while decoding MB 58 3, bytestream -34)
# input_str= 'rtsp://admin:yxgl$666@192.168.200.203:554'

# cameara is not online
# input_str= 'rtsp://admin:yxgl$666@192.168.200.200:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.201:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.206:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.207:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.208:554'


# input_str= 'rtsp://admin:yxgl$666@192.168.200.195:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.196:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.199:554'
# input_str= 'rtsp://admin:yxgl$666@192.168.200.204:554'

input_str= 'rtsp://admin:yxgl$666@192.168.200.205:554'


#rtsp://admin:yxgl!123456@10.249.181.9:554/Streaming/Channels/101
# 大华
# input_str = 'rtsp://admin:admin123@10.248.10.100:554/cam/realmonitor?channel=1&subtype=0'
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