save_dir_name = "v1_debug"
config = {
    #####set detect image in scenes chanel_IP #####
    "scene_jacket": [
        '0',
        '1',
        '2',
        '3'

	 ],

    #####Queues#####
    'mq_username': 'test',
    'mq_pswd': 'test',  
    'mq_server_host': '127.0.0.1',
    'mq_server_port': '5672',
    'frame_q_in': 'hello',
    'frame_q_out':'hx_jacket_out',
}

