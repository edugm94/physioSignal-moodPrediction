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

    def __cleanDataset(self, data_, label_):
        # Obtain an accounting of hbiw many vectors there is for each emotion
        unique, counts = np.unique(label_, return_counts=True)
        counting = dict(zip(unique, counts))

        # Get the total amount of vectors and the threshold to filter dictionary
        tot = sum(counting.values())
        threshold = tot * 0.1

        # Get a dictionary with the emotions that should be cleaned from the initial variables
        # It is kept a dictionary to check the lenght of the cleaned values at the end
        emo_del_dict = dict(filter(lambda elem: elem[1] < threshold, counting.items()))
        # Array that contains the value of the emotions to be cleaned in the "labels" variable
        emo_del_arr = np.array(list(emo_del_dict.keys()))

        # Array containing the index that should be deleted from "data" and "label"
        indx_del_arr = np.where(label_ == emo_del_arr)[0]
        assert indx_del_arr.shape[0] == sum(
            emo_del_dict.values()), "The amount of vectors to delete does not match! Check it."

        # Clean the final array according to the found indexes
        labels_clean = np.delete(label_, indx_del_arr).reshape(-1, 1)
        data_clean = dict()
        for k, v in data_.items():
            data_clean[k] = np.delete(v, indx_del_arr, axis=0)

        return data_clean, labels_clean

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

            # When a day has -1 as raw_vectors means that there was some problem while extracting the vectors
            if acc_ds.shape == (1, 1):
                continue

            if (acc_ds.shape[0] == hr_ds.shape[0] == eda_ds.shape[0] == temp_ds.shape[0]):
                pass
            else:
                num_vec = min(acc_ds.shape[0], hr_ds.shape[0], eda_ds.shape[0], temp_ds.shape[0])
                acc_ds = acc_ds[0:num_vec, :, : ]
                eda_ds = eda_ds[0:num_vec, :]
                hr_ds = hr_ds[0:num_vec, :]
                temp_ds = temp_ds[0:num_vec, :]

            # It is extracted the label assigned for each signal to check they coicinde
            # TODO: It may be modified the code when an error occur!
            acc_l = acc_ds.item((0, 0, 0))
            hr_l = hr_ds.item((0, 0))
            eda_l = eda_ds.item((0, 0))
            temp_l = temp_ds.item((0, 0))

            if acc_l == hr_l == eda_l == temp_l:
                # First column must be taken and stored in label_ array

                label = hr_ds[:, 0].astype(int).reshape(-1, 1)
                label_ = label if len(label_) == 0 else np.concatenate((label_, label), axis=0)

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

            # Third channel is added in order not to crush with Conv1D layer from TensorFlow
            hr_ = hr_.reshape(-1, hr_.shape[1], 1)
            eda_ = eda_.reshape(-1, eda_.shape[1], 1)
            temp_ = temp_.reshape(-1, temp_.shape[1], 1)

            data_ = {
                "acc": acc_,
                "eda": eda_,
                "hr": hr_,
                "temp": temp_
            }

        # Clean the less representative emotions captured by the smartwatch
        data_, label_ = self.__cleanDataset(data_, label_)
        dataset = tf.data.Dataset.from_tensor_slices((data_, label_))

        return dataset
