save_dir_name = "v1_debug"
config = {
    #####set detect image in scenes chanel_IP #####
    "scene_belt": [
        '0',
        '1',
        '2',
        '3'
	 ],

    #####Queues#####
    'mq_username': 'test',
    'mq_pswd': 'test',  
    # 'mq_server_host': '127.0.0.1',
    'mq_server_host': '10.248.68.249',
    'mq_server_port': '5672',
    'frame_q_in': 'SafetyBelt',
    'frame_q_out':'hx_safetybelet_out',
}

