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

        for day in self.hdf5.keys():
            acc = self.hdf5[day + '/acc'].get('raw_vectors')[()]
            hr = self.hdf5[day + '/hr'].get('raw_vectors')[()]
            eda = self.hdf5[day + '/eda'].get('raw_vectors')[()]
            temp = self.hdf5[day + '/temp'].get('raw_vectors')[()]

            acc_l = acc.item((0, 0, 0))
            hr_l = hr.item((0, 0))
            eda_l = eda.item((0, 0))
            temp_l = temp.item((0, 0))

            if acc_l == hr_l == eda_l == temp_l:
                label = int(acc_l)
            else:
                print("Error! Labels do not coincide.")

            # Raw vectors after extracting the label
            acc_x = np.delete(acc, [0], axis=1)
            hr_x = np.delete(hr, [0], axis=1)
            eda_x = np.delete(eda, [0], axis=1)
            temp_x = np.delete(temp, [0], axis=1)