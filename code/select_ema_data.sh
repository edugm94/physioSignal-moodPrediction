#!/bin/sh

clear;

NUM_PATIENTS=2
SAMPLING_DAYS="5 5"
WINDOW_SIZE=60
LABEL="mood"
PATH_DATA="data/datos_E4/"
OUTPUT_PATH="out/1_ema_data/"
ALL=1

if [[ $1 == '-help' ]]
then
  python3 select_ema_data.py --help
else
  echo "
 ________  ___  ___   ______      _
|  ___|  \/  | / _ \  |  _  \    | |
| |__ | .  . |/ /_\ \ | | | |__ _| |_ __ _
|  __|| |\/| ||  _  | | | | / _\` | __/ _\` |
| |___| |  | || | | | | |/ / (_| | || (_| |
\____/\_|  |_/\_| |_/ |___/ \__,_|\__\__,_|


 _____       _ _           _   _
/  __ \     | | |         | | (_)
| /  \/ ___ | | | ___  ___| |_ _  ___  _ __
| |    / _ \| | |/ _ \/ __| __| |/ _ \| '_ \\
| \__/\ (_) | | |  __/ (__| |_| | (_) | | | |
 \____/\___/|_|_|\___|\___|\__|_|\___/|_| |_|"

  python3 select_ema_data.py --num_patients $NUM_PATIENTS --sampling_days  $SAMPLING_DAYS --window_size $WINDOW_SIZE \
                --label $LABEL --data_path $PATH_DATA --output_path $OUTPUT_PATH --all $ALL
fi


