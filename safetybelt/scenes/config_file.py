save_dir_name = "v1_debug"
config = {
    #####set detect image in scenes chanel_IP #####
    "scene_belt": [
        '192.168.1.1_1',
	 ],

    #####Queues#####
    'mq_username': 'liujin',
    'mq_pswd': '123456',  
    'mq_server_host': '127.0.0.1',
    'mq_server_port': '5672',
    'frame_q_in': 'frame_in',
    'frame_q_out':'frame_out',
}

