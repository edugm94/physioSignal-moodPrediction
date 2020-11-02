#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
#   Author: Eduardo Gutierrez Maestro
#   Date: 2020.09.18
#   email: eduardo.gutierrez-maestro@oru.se
#
#   Center for Applied Autonomous Sensor Systems (AASS), Cognitive Robotic Systems Labs
#   University of Orebro, Sweden
#
from utils.CreateRAWData import CreateRAWData

path_to_ema = 'out/1_ema_data/p1_ema_mood_60.h5'
output_path = 'out/2_raw_data/'
overlapping = 10
w_feat_size = 60

raw = CreateRAWData(
    path_to_hdf5_ema=path_to_ema,
    output_path=output_path,
    overlapping=overlapping,
    window_feat_size=w_feat_size
)

raw.createRawVector()



