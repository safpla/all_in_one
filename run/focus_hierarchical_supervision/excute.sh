export CUDA_VISIBLE_DEVICES=3
nohup python run.py config1.2.ini 1 >> log/nohup_config1.2_1.txt &

export CUDA_VISIBLE_DEVICES=2
nohup python run.py config1.3.ini 1 >> log/nohup_config1.3_1.txt &

export CUDA_VISIBLE_DEVICES=1
nohup python run.py config1.4.ini 1 >> log/nohup_config1.4_1.txt &

export CUDA_VISIBLE_DEVICES=0
nohup python run.py config1.5.ini 1 >> log/nohup_config1.5_1.txt &
