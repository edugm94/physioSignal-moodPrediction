#!/bin/sh
#
#   Author: Eduardo Gutierrez Maestro
#   Date: 2020.09.18
#   email: eduardo.gutierrez-maestro@oru.se
#
#   Center for Applied Autonomous Sensor Systems (AASS), Cognitive Robotic Systems Labs
#   University of Orebro, Sweden
#
NUM_PATIENTS=2
SAMPLING_DAYS="1 1"
WINDOW_SIZE=60
LABEL="mood"
PATH_DATA="data/datos_E4/"
OUTPUT_PATH="out/"
OUTPUT_NAME="p1_mood_60.h5"
#TYPE=0


python build_up_dataset.py --num_patients $NUM_PATIENTS --sampling_days  $SAMPLING_DAYS --window_size $WINDOW_SIZE \
							--label $LABEL --data_path $PATH_DATA --output_path $OUTPUT_PATH --output_name $OUTPUT_NAME
