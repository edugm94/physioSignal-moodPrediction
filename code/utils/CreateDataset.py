#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import pandas as pd
import os
import h5py
from utils.SignalProcessing import SignalProcessing


class CreateDataset():

    def __init__(self, num_patients, sampling_days, window_size, type_label, path_to_csv, output_path, output_filename):
        if not os.path.exists(path_to_csv):
            raise NameError('Wrong path!\n'
                            'Input a existing datafile path, please.')
        if not os.path.exists(output_path):
            raise NameError('Wrong path!\n'
                            'Input a existing output datafile path, please.')
        assert num_patients == len(sampling_days), 'Sampling days array length must be equal' \
                                                      'as the number of input patients, check it please.'
        self.num_patients = num_patients
        self.sampling_days = sampling_days
        self.path_to_csv = path_to_csv
        self.output_path = output_path
        self.output_filename = output_filename
        self.signal_types = ['acc', 'eda', 'hr', 'temp']
        self.ws = window_size
        self.label = type_label
        self.hdf5_obj = None

    def __createHDF5Object(self):
        self.hdf5_obj = h5py.File(self.output_path + self.output_filename, 'w')

    def __closeHDF5Oject(self):
        self.hdf5_obj.close()

    def createDataset(self):

        """
        This method creates for every patient, signal, and day
        :return:
        """

        for num_patient in range(self.num_patients):
            data_path = self.path_to_csv + 'P' + str(num_patient + 1) + '/P' + str(num_patient + 1) + '_Complete/'
            
            if not os.path.exists(data_path):
                raise NameError('There is no data path!\n'
                                'Check if the attributes are correct, please.')

            for signal in self.signal_types:
                signal_path = data_path + signal.upper() + str(day + 1) + '.csv'

                for day in range(self.sampling_days[num_patient]):
                    ema_path = data_path + "EMAs" + str(num_patient + 1) + '.xlsx'


                    sp = SignalProcessing(
                        type_signal=signal,
                        path_to_file=signal_path,
                        path_to_ema=ema_path,
                        window_size=self.ws,
                        type_label=self.label

                    )

                    vectors, labels = sp.getGroundTruth()

                    self.hdf5_obj.create_dataset(signal + "/" + "day" + day + "/vectors", data=vectors)
                    self.hdf5_obj.create_dataset(signal + "/" + "day" + day + "/labels", data=labels)

        self.__closeHDF5Oject()





        
                    #print("Signal type processed: {} -- Day : {}".format(signal, day+1))
                    #print(arr_vector.shape)
                    #print(arr_label.shape)




                    #print(signal_path)
                    #print("Patient num {} --> Day : {} --> Signal {}".format(
                    #    str(num_patient + 1), str(day + 1), signal))






