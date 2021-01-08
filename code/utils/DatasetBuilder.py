#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
#   Author: Eduardo Gutierrez Maestro
#   Date: 2021.01.08
#   email: eduardo.gutierrez-maestro@oru.se
#
#   Center for Applied Autonomous Sensor Systems (AASS), Cognitive Robotic Systems Labs
#   University of Orebro, Sweden
#
import h5py
import numpy as np
import tensorflow as tf

class DatasetBuilder:
    def __init__(self, path_to_hdf5):
        self.path_hdf5 = path_to_hdf5

    def __readHDF5(self):
        self.hdf5 = h5py.File(self.path_hdf5, 'r')

    def buildDataset(self):
        # TODO: Possible idea! It can be passed as well a list of files in case it is desired to create a TF dataset
        #  file that contains several patients --> It should be study after defining the kind of experiments to be done.
        self.__readHDF5()

        # Auxiliary arrays which will for part of the dictionary to set up the TF dataset object
        acc_ = np.array([])
        hr_ = np.array([])
        eda_ = np.array([])
        temp_ = np.array([])
        label_ = np.array([])

        for day in self.hdf5.keys():
            # Load the different dataset for each signal
            acc_ds = self.hdf5[day + '/acc'].get('raw_vectors')[()]
            hr_ds = self.hdf5[day + '/hr'].get('raw_vectors')[()]
            eda_ds = self.hdf5[day + '/eda'].get('raw_vectors')[()]
            temp_ds = self.hdf5[day + '/temp'].get('raw_vectors')[()]

            # It is extracted the label assigned for each signal to check they coicinde
            # TODO: It may be modified the code when an error occur!
            acc_l = acc_ds.item((0, 0, 0))
            hr_l = hr_ds.item((0, 0))
            eda_l = eda_ds.item((0, 0))
            temp_l = temp_ds.item((0, 0))

            if acc_l == hr_l == eda_l == temp_l:
                label = int(acc_l)
            else:
                print("Error! Labels do not coincide.")

            # Raw vectors after extracting the label
            acc_x = np.delete(acc_ds, [0], axis=1)
            hr_x = np.delete(hr_ds, [0], axis=1)
            eda_x = np.delete(eda_ds, [0], axis=1)
            temp_x = np.delete(temp_ds, [0], axis=1)

            # TODO: I am assuming that all vectors have same lenght. In case two consecutive EMAs are too close, the
            #  lenght of the vector can be shorter. It should be done something about this: either pad the vector or
            #  discard the vector. ATTENTION!

            acc_ = acc_x if len(acc_) == 0 else np.concatenate((acc_, acc_x), axis=0)
            hr_ = hr_x if len(hr_) == 0 else np.concatenate((hr_, hr_x), axis=0)
            eda_ = eda_x if len(eda_) == 0 else np.concatenate((eda_, eda_x), axis=0)
            temp_ = temp_x if len(temp_) == 0 else np.concatenate((temp_, temp_x), axis=0)


