#!/bin/bash

clear;

OUT_PATH='out/2_raw_data/'
PATH_TO_HDF5_EMA='out/1_ema_data/'
OVERLAPPING=10
WINDOW_FEAT_SIZE=60
LABEL='mood'
ALL=0


if [[ $1 == '-help' ]]
then
  python3 select_ema_data.py --help
else

  echo "______                                _
| ___ \                              | |
| |_/ /__ ___      __ __   _____  ___| |_ ___  _ __
|    // _\` \ \ /\ / / \ \ / / _ \/ __| __/ _ \| '__|
| |\ \ (_| |\ V  V /   \ V /  __/ (__| || (_) | |
\_| \_\__,_| \_/\_/     \_/ \___|\___|\__\___/|_|


                                 _
                                | |
  __ _  ___ _ __   ___ _ __ __ _| |_ ___  _ __
 / _\` |/ _ \ '_ \ / _ \ '__/ _\` | __/ _ \| '__|
| (_| |  __/ | | |  __/ | | (_| | || (_) | |
 \__, |\___|_| |_|\___|_|  \__,_|\__\___/|_|
  __/ |
 |___/                                      "

  python3 build_raw_dataset.py --out_path $OUT_PATH --path_to_hdf5_ema $PATH_TO_HDF5_EMA --overlapping $OVERLAPPING \
          --window_feat_size $WINDOW_FEAT_SIZE --label $LABEL --all $ALL
fi

