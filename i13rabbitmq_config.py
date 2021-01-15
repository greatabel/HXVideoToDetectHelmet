from os import environ


FactoryName = environ.get('FactoryName', '华新水泥')

QUEUE_SIZE = 5
ARGUMENTS = {'x-max-length': QUEUE_SIZE, "x-queue-mode": "lazy"}

'''
AI_SERVER_2 is the 2gpu server with belt/jacket
AI_SERVER_1 is the 2gpu server only hamlet
AI_SERVER_0 is the 1gpu server only hamlet
VM_0 is the virtual machine , no gpu

'''
# placeid_bound_dict = {
# 	'AI_SERVER_2': (0, 20),
# 	'AI_SERVER_1': (21, 61),
# 	'AI_SERVER_0': (62, 75),
# 	'VM_0': (21, 61),
# }
placeid_bound_dict = {
	'AI_SERVER_2': (0, 42),
	'AI_SERVER_1': (43, 94),
	'AI_SERVER_0': (95, 122),
	'VM_0': (43, 94),
}

AI_SERVER_NUMBER = environ.get('AI_SERVER_NAME')
AI_SERVER_NUMBER_placeid_start = placeid_bound_dict[AI_SERVER_NUMBER][0]
AI_SERVER_NUMBER_placeid_end = placeid_bound_dict[AI_SERVER_NUMBER][1]

# set the rabbit-server, local ai-server need to get data from
servernumber_rabbitIP = {
	'AI_SERVER_2': '10.248.68.59',
	'AI_SERVER_1': '10.248.68.244',
	'AI_SERVER_0': '10.248.68.203',
	'VM_0': '10.248.68.249'
}


AI_SERVER_RABBIT_IP = servernumber_rabbitIP[AI_SERVER_NUMBER]
VM_0_RABBIT_IP = servernumber_rabbitIP['VM_0']


Where_This_Server_ReadFrom = AI_SERVER_RABBIT_IP
# 临时因为ai-1 要处理最大路数，暂时不在这台机器上读取视频流，从机架虚拟机读取 
if AI_SERVER_NUMBER == "AI_SERVER_1":
	Where_This_Server_ReadFrom = VM_0_RABBIT_IP


print('FactoryName=', FactoryName,
	  'AI_SERVER_NUMBER_placeid_start=', AI_SERVER_NUMBER_placeid_start,
	  'AI_SERVER_NUMBER_placeid_end=', AI_SERVER_NUMBER_placeid_end,
	  'AI_SERVER_RABBIT_IP=', AI_SERVER_RABBIT_IP,
	  'VM_0=', VM_0_RABBIT_IP,
	  'Where_This_Server_ReadFrom=', Where_This_Server_ReadFrom)
