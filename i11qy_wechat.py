'''
新建了一个应用

https://work.weixin.qq.com/api/doc/90000/90135/90236

https://work.weixin.qq.com/api/doc/90000/90135/91039

HX_QY_WECHAT_CORPID
HX_QY_WECHAT_CORPSECRET

'''

import requests
from os import environ
import json
import simplejson


qy_wechat_corpid = environ.get('HX_QY_WECHAT_CORPID')
qy_wechat_corpsecret = environ.get('HX_QY_WECHAT_CORPSECRET')
print(qy_wechat_corpid, qy_wechat_corpsecret)