# HXVideoToDetectHelmet

on osx, if the camera is not accessible: sudo killall VDCAssistant

2020_11_02 
at inpout:
virtual_env enter: sdk_reader
then run: 
python3 i13multiple_sdk_temp.py

at process
then run: 
python3 i13multiple_processor_obj.py --gpu=True --network=yolo3_mobilenet0.25_voc



2020_11_04
at inpout:
virtual_env enter: sdk_reader
python3 i14send_to_multiple_channels.py

at process
virtual_env enter: samaritan0
then run: 
python3 i13multiple_processor_obj.py --gpu=True --network=yolo3_mobilenet0.25_voc
python3 module_belt_mq.py 0
python3 module_jacket_mq.py 
