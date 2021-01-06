#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
#   Author: Eduardo Gutierrez Maestro
#   Date: 2020.10.16
#   email: eduardo.gutierrez-maestro@oru.se
#
#   Center for Applied Autonomous Sensor Systems (AASS), Cognitive Robotic Systems Labs
#   University of Orebro, Sweden
#

import h5py
import os
from tqdm import tqdm
from utils.SlidingWindow import SlidingWindow
import numpy as np

class CreateRAWData:
    def __init__(self, output_path, path_to_hdf5_ema, overlapping, window_feat_size, label, all):

        self.output_path = output_path + label + '/'
        self.path_to_hdf5_ema = path_to_hdf5_ema + label + '/'
        self.overlapping = overlapping
        self.window_feat_size = window_feat_size
        self.label = label
        self.all = all

        if self.all == 0:
            self.output_path = self.output_path + 'individual/'
            self.path_to_hdf5_ema = self.path_to_hdf5_ema + 'individual/'
        else:
            self.output_path = self.output_path + 'all/'
            self.path_to_hdf5_ema = self.path_to_hdf5_ema + 'all/'

        self.signaL_freq_dict = {"acc": 32,
                            "temp": 4,
                            "eda": 4,
                            "hr": 1}

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def __readHDF5(self, filename):
        self.in_h5 = h5py.File(self.path_to_hdf5_ema + filename, 'r')

    def __createOutHDF5(self, filename):
        self.out_h5 = h5py.File(self.output_path + filename, 'w')

    def __closeHDF5Objects(self):
        self.in_h5.close()
        self.out_h5.close()

    def createRawVector(self):
        # Initially we read the HDF5 containing the EMA data points
        # and create the object file for the output HDF5

        input_files = sorted(os.listdir(self.path_to_hdf5_ema))
        num_patients = len(input_files)

        for patient_id, file in enumerate(tqdm(input_files)):
            self.__readHDF5(file)

            # Create name and object to save output file
            out_filename = "p" + str(patient_id + 1) + "_raw_" + self.label + '.h5' if self.all == 0 \
                else "p" + str(patient_id + 1) + "_raw_" + self.label + '_all.h5'
            self.__createOutHDF5(out_filename)

            tbar = tqdm(enumerate(self.in_h5.keys()), position=0)
            for day_idx, day in tbar:
                tbar.set_description('Patient {} | day {}: '.format(patient_id + 1, day_idx + 1))

                for signal in self.in_h5[day].keys():
                    dataset = self.in_h5[day + '/' + signal]
                    vectors = dataset.get('vectors')[()]
                    labels = dataset.get('labels')[()]

                    #if vectors.shape[1] == 1 and labels.shape[1] == 1:
                    if vectors.shape[1] and labels.shape[1] == 1:
                        # In case of an error detected from input HDF5 file, set -1 in the output file
                        # indicating that there is a mistake
                        self.out_h5.create_dataset(name=day + '/' + signal + "/raw_vectors", data=np.array([[-1]]))
                    else:
                        slideW_obj = SlidingWindow(
                            vectors=vectors,
                            labels=labels,
                            overlapping=self.overlapping,
                            samp_freq=self.signaL_freq_dict[signal],
                            window_feat_size=self.window_feat_size,
                            signal_type=signal
                        )
                        raw_vector_matrix = slideW_obj.extractRawVector()

                        # create data set containing the matrix of raw vectors
                        self.out_h5.create_dataset(name=day + '/' + signal + "/raw_vectors", data=raw_vector_matrix)

            # Close objects
            self.__closeHDF5Objects()









        '''
        self.__readHDF5()
        self.__createOutHDF5()

        for day_idx, day in enumerate(self.in_h5.keys()):
            for signal in self.in_h5[day].keys():


                dataset = self.in_h5[day + '/' + signal]
                vectors = dataset.get('vectors')[()]
                labels = dataset.get('labels')[()]

                slideW_obj = SlidingWindow(
                    vectors=vectors,
                    labels=labels,
                    overlapping=self.overlapping,
                    samp_freq=self.signaL_freq_dict[signal],
                    window_feat_size=self.window_feat_size,
                    signal_type=signal
                )
                raw_vector_matrix = slideW_obj.extractRawVector()

                # create data set containing the matrix of raw vectors
                self.out_h5.create_dataset(name=day + '/' + signal + "raw_vectors", data=raw_vector_matrix)

        self.__closeHDF5Objects()
        '''
