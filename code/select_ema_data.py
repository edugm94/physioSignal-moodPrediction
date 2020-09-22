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
from utils.SelectEMAData import SelectEMAData

def argParser():
    parser = argparse.ArgumentParser(description="This script's target is to set up a dataset by creating "
                                                 "HDF5 files containing the raw signals with its corresponding labels"
                                                 " for each EMA answered with a specific window size.")
    parser.add_argument('-n_p', '--num_patients', help='Number of patients containing the dataset', default=1, type=int)
    parser.add_argument('-s_d', '--sampling_days', help='List of numbers. Express the number of available sampling days'
                                                        ' for each patient. The index in the list corresponds to each '
                                                        'patient. E.g: [4, 5, 3] --> Patient 1 (4 sampling days); '
                                                        'patient 2 (5 sampling days); patient 3 (3 sampling days) and '
                                                        'so on.', nargs='+', default=["2"])
    parser.add_argument('-ws', '--window_size', help='Time size by which EMA will be chunked.', type=int, default=60)
    parser.add_argument('-l', '--label', help='The label which that you want to build up the dataset.', default='mood',
                        type=str)
    parser.add_argument('-p_dat', '--data_path', help='The path where the .CSV files are located.',
                        default='/data/datos_E4/', type=str)
    parser.add_argument('-p_out', '--output_path', help='Path where you want to save the HDF5 files.',
                        default='out/', type=str)
    parser.add_argument('-n_out', '--output_name', help='Name of the dataset: HDF5 file.', default='p1_mood_60.h5',
                        type=str)
    parser.add_argument('-t', '--type', help='Type of dataset: unique patient or multi patient. Codes:'
                                             ' 0 for unique patient dataset; 1 for all patients dataset', default=0,
                        type=int)

    args = parser.parse_args()
    return args


def main():
    args = argParser()

    num_patients = args.num_patients
    sampling_days = [int(i) for i in args.sampling_days]
    window_size = args.window_size
    label = args.label
    path_data = args.data_path
    output_path = args.output_path
    output_name = args.output_name
    #type = args.type



    dataset_obj = SelectEMAData(
        num_patients=num_patients,
        sampling_days=sampling_days,
        path_to_csv=path_data,
        output_path=output_path,
        output_filename=output_name,
        window_size=window_size,
        type_label=label
    )

    dataset_obj.selectEMAData()

if __name__ == '__main__':
    main()