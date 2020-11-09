save_dir_name = "v1_debug"
config = {
    #####set lifejacket scenes #####
    "scene_jacket": [
        {'1':[(0,80,140),(30,255,255),(100,80,30),(124,255,255)]},
        {'2':[(0,80,140),(10,255,255),(100,43,30),(124,255,255)]},
      
        {'default': [(0,80,140),(30,255,255),(100,80,30),(124,255,255)]}
    ],
    #####set safetybelt scenes #####
    "scene_belt": [
        'default_belt',
        '3',
        '4',
        '5'
	 ],

    #####Queues#####
    'mq_username': 'test',
    'mq_pswd': 'test',  
    # 'mq_server_host': '10.248.68.249',
    'mq_server_host': '127.0.0.1',
    'mq_server_port': '5672',
    'lifeJacket_q_in': 'LifeJacket',
    # 'lifeJacket_q_out':'lifeJacket_out',
    'safetyBelt_q_in': 'SafetyBelt',
    # 'safetyBelt_q_out':'safetyBelt_out',
}

