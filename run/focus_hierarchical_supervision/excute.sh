export CUDA_VISIBLE_DEVICES=3
nohup python run.py config1.1.ini _data0125 >> log/nohup_config1.1_data0125.txt &

export CUDA_VISIBLE_DEVICES=0
nohup python run.py config3.1.ini _data0125 >> log/nohup_config3.1_data0125.txt &
#
#export CUDA_VISIBLE_DEVICES=3
#nohup python run.py config1.4.ini 1 >> log/nohup_config1.4_1.txt &
#
#export CUDA_VISIBLE_DEVICES=0
#nohup python run.py config1.5.ini 1 >> log/nohup_config1.5_1.txt &
