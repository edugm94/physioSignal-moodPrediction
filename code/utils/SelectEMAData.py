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
import os
import h5py
from utils.SignalProcessing import SignalProcessing
from tqdm import tqdm


class SelectEMAData:
    """
    This class objective is to go through all patients files in a iterative way (for-loop).
    Within this class it will be called to the class SignalProcessing() to process every .CSV file
    build-up a training dataset.
    The user will input some information regarding how many patients is needed to be processed and the number of
    sampling days.
    """
    def __init__(self, num_patients, sampling_days, window_size, type_label, path_to_csv, output_path, all):
        if not os.path.exists(path_to_csv):
            raise NameError('Wrong path!\n'
                            'Input a existing datafile path, please.')


        assert num_patients == len(sampling_days), 'Sampling days array length must be equal' \
                                                      'as the number of input patients, check it please.'
        self.num_patients = num_patients
        self.sampling_days = sampling_days
        self.path_to_csv = path_to_csv
        self.output_path = output_path + type_label + '/'
        if all == 0:
            self.output_path = self.output_path + 'individual/'
        else:
            self.output_path = self.output_path + 'all/'
        self.all = all      # Indicates if used all patients data or just one patient (signals normalization)
        self.signal_types = ['acc', 'eda', 'hr', 'temp']
        self.ws = window_size
        self.label = type_label
        self.hdf5_obj = None
        # Create always a new directory if it does not exist
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def __createHDF5Object(self, num_patient):
        if self.all == 1:
            self.hdf5_obj = h5py.File(self.output_path +
                                      'p' + str(num_patient + 1) + '_ema_' + self.label + '_' + str(self.ws) +
                                      '_all.h5', 'w')
        else:
            self.hdf5_obj = h5py.File(self.output_path +
                                      'p' + str(num_patient+1) + '_ema_' + self.label + '_' + str(self.ws) + '.h5', 'w')

    def __closeHDF5Oject(self):
        self.hdf5_obj.close()

    def selectEMAData(self):

        """
        This method creates a dataset containing significant EMA data with its corresponding label.
        The output file has the following structure: group/group/dataset,
        e.g: acc/day1/vectors or acc/day1/labels. Each signal type is a group and each day is a subgroup
        within each signal type. This architecture will help to organize the data.
        The lower level in the file  hierarchy is composed byt two datasets:
            · Vectors: Contains the raw filtered signals. Multidimensional array
            · Labels: Contains the label for a specific emotion (mood, happiness or arousal) for each array in
                    Vectors Dataset.

        :return: A HDF5 file which will be named as "output_filename" and stored in "output_path"
        """



        for num_patient in range(self.num_patients):
            data_path = self.path_to_csv + 'P' + str(num_patient + 1) + '/P' + str(num_patient + 1) + '_Complete/'
            # Open HDF5 object to store data (one for each patient)
            self.__createHDF5Object(num_patient)

            if not os.path.exists(data_path):
                raise NameError('There is no data path!\n'
                                'Check if the attributes are correct, please.')

            with tqdm(range(self.sampling_days[num_patient]), position=0) as t:
                for day in t:
                    t.set_description('Patient {} | day {}: '.format(num_patient+1, day+1))
                    ema_path = data_path + "EMAs" + str(day + 1) + '.xlsx'


                    for signal in self.signal_types:
                        signal_path = data_path + signal.upper() + str(day + 1) + '.csv'

                        # Call SignalProcessing Class to obtain vectors and labels for a certain emotion
                        # with a fixed window size

                        sp = SignalProcessing(
                            type_signal=signal,
                            path_to_file=signal_path,
                            path_to_ema=ema_path,
                            window_size=self.ws,
                            type_label=self.label,
                            all=self.all
                        )

                        vectors, labels = sp.getGroundTruth()

                        # Create for each type of signal and day: two datasets containing both: labels and raw signals
                        #self.hdf5_obj.create_dataset(signal + "/" + "day" + str(day + 1) + "/vectors", data=vectors)
                        #self.hdf5_obj.create_dataset(signal + "/" + "day" + str(day + 1) + "/labels", data=labels)
                        self.hdf5_obj.create_dataset("day" + str(day + 1) + '/' + signal + "/vectors", data=vectors)
                        self.hdf5_obj.create_dataset("day" + str(day + 1) + '/' + signal + "/labels", data=labels)
        # Close HDF5 file object one is finished the iteration
        self.__closeHDF5Oject()










