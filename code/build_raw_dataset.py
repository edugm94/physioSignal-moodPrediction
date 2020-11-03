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
import argparse
from utils.CreateRAWData import CreateRAWData


def argParser():
    parser = argparse.ArgumentParser(description="This script is used to create the HDF5 files containing"
                                                 "the raw vector with their correspondig labels.")
    parser.add_argument('-out_path', '--out_path', help='Path where HDF5 files with raw vectors will be stored.',
                        type=str, default='out/2_raw_data/')
    parser.add_argument('-path_hdf5', '--path_to_hdf5_ema', help='Path where are located the HDF5 files with'
                                                                 'the corresponding EMA data points.',
                        type=str, default='out/1_ema_data/')
    parser.add_argument('-ov', '--overlapping', help='Percentage of overlapping when creating the raw vectors.',
                        type=int, default=10)
    parser.add_argument('-w_feat', '--window_feat_size', help='Size of the sliding window to form the raw vector.'
                                                             'Expressed in seconds (s).', type=int, default=60)
    parser.add_argument('-lab', '--label', help='Label that want to be processed. Options: mood, hapinnes or '
                                                'arousal', type=str, default='mood')
    parser.add_argument('-all', '--all', help='Flag to indicate if it is desired to obtain raw vectors in a'
                                              'individial or grupal fashion. 0 = individual; 1 = grupal',
                        type=int, default=0)

    args = parser.parse_args()
    return args


def main():

    args = argParser()
    output_path = args.out_path
    path_to_hdf5_ema = args.path_to_hdf5_ema
    overlapping = args.overlapping
    window_feat_size = args.window_feat_size
    label = args.label.lower()
    all = args.all

    raw_data_obj = CreateRAWData(
        output_path=output_path,
        path_to_hdf5_ema=path_to_hdf5_ema,
        overlapping=overlapping,
        window_feat_size=window_feat_size,
        label=label,
        all=all
    )

    raw_data_obj.createRawVector()



if __name__ == '__main__':
    main()



