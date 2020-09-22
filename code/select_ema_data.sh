#!/bin/sh

NUM_PATIENTS=2
SAMPLING_DAYS="3 3"
WINDOW_SIZE=60
LABEL="mood"
PATH_DATA="data/datos_E4/"
OUTPUT_PATH="out/1_ema_data/"
#OUTPUT_NAME="p1_ema_mood_60.h5"
#TYPE=0




python3 select_ema_data.py --num_patients $NUM_PATIENTS --sampling_days  $SAMPLING_DAYS --window_size $WINDOW_SIZE \
							--label $LABEL --data_path $PATH_DATA --output_path $OUTPUT_PATH
							#--output_name $OUTPUT_NAME
