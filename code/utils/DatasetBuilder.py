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

class DatasetBuilder:
    def __init__(self, path_to_hdf5):
        self.path_hdf5 = path_to_hdf5

    def __readHDF5(self):
        self.hdf5 = h5py.File(self.path_hdf5, 'r')

    def createTFDataset(self):
        self.__readHDF5()

        print('holla')