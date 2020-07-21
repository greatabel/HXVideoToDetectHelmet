'''
新建了一个应用

https://work.weixin.qq.com/api/doc/90000/90135/90236

https://work.weixin.qq.com/api/doc/90000/90135/91039

HX_QY_WECHAT_CORPID
HX_QY_WECHAT_CORPSECRET

https://work.weixin.qq.com/api/doc/90000/90135/90253

'''

import requests
from os import environ
import json
import simplejson
from requests_toolbelt import MultipartEncoder


filepath = 'screenshots/'
filename = '1_2020-07-17-17_16_40.jpg'
msg = "阳新骨料ai测试"
test_touser = "80002302|00038910"

# 在ubuntu中配置环境变量在：  ~/.bashrc
qy_wechat_corpid = environ.get('HX_QY_WECHAT_CORPID')
qy_wechat_corpsecret = environ.get('HX_QY_WECHAT_CORPSECRET')

agentid = 1000030
print('get qy_wechat_corpid qy_wechat_corpsecret from os:', qy_wechat_corpid, qy_wechat_corpsecret)

def get_token():
  url_get_token = 'https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={}&corpsecret={}'\
                      .format(qy_wechat_corpid, qy_wechat_corpsecret)
  r = requests.get(url_get_token)
  d = json.loads(r.text)
  print('d=', d)
  return d['access_token']


def upload_media_to_wechat(access_token, filename):
  post_file_url = "https://qyapi.weixin.qq.com/cgi-bin/media/upload?access_token={}&type=file"\
                      .format(access_token)
  m = MultipartEncoder(
      fields={filename: ('image', open(filepath + filename, 'rb'), 'image/jpeg')},
  )
  print(m)
  r = requests.post(url=post_file_url, data=m, headers={'Content-Type': m.content_type})
  image_d = json.loads(r.text)
  print('image_d=', image_d)
  return image_d['media_id']



def send_text_and_image_wechat(filename, msg, touser):

  access_token = get_token()
  media_id = upload_media_to_wechat(access_token, filename)
  print('-'*20)
  print('msg=', msg)
  url_get_send = 'https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={}'\
                      .format(access_token)
  send_msg_json = {
     "touser" : touser,

     "msgtype" : "text",
     "agentid" : agentid,
     "text" : {
         "content" : msg
     },
     "safe":0
  }
  json_string = simplejson.dumps(send_msg_json, ensure_ascii=False).encode('utf8')
  headers = {'Content-Type': 'application/json', "charset": "utf-8"}

  r1 = requests.post(url_get_send, data=json_string, headers=headers)
  print('send text r1.text = ', r1.text)


  send_img_json = {
     "touser" : touser,

     "msgtype" : "image",
     "agentid" : agentid,
     "image" : {
          "media_id" : media_id
     },
     "safe":0
  }
  json_string = simplejson.dumps(send_img_json, ensure_ascii=False).encode('utf8')
  headers = {'Content-Type': 'application/json', "charset": "utf-8"}

  r2 = requests.post(url_get_send, data=json_string, headers=headers)
  print('send image r2.text = ', r2.text)


if __name__ == "__main__":
  send_text_and_image_wechat(filename, msg, touser=test_touser)