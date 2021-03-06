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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class DatasetBuilder:
    def __init__(self, path_to_hdf5, leave_one_out, one_hot):
        self.path_hdf5 = path_to_hdf5
        self.leave_one_out = leave_one_out
        self.one_hot = one_hot

    def __readHDF5(self):
        self.hdf5 = h5py.File(self.path_hdf5, 'r')

    def __cleanDataset(self, data_, label_):
        '''This method is used to get rid of meaningless emotions'''
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

    def __oneHotLabels(self, labels):
        le = LabelEncoder()
        le.fit(labels)
        labels_new = le.transform(labels)
        labels_oh = tf.keras.utils.to_categorical(labels_new)

        return labels_oh

    def __splitTrainTest(self, data, labels):
        '''This method aims to split dataset into train and test set.
        The proportions are 75% and 25% according to the paper.'''
        ratio = 0.25
        seed = 42

        accTrainData, accTestData, accTrainLabel, accTestLabel = train_test_split(data["acc"], labels,
                                                                                  test_size=ratio, random_state=seed)
        edaTrainData, edaTestData, edaTrainLabel, edaTestLabel = train_test_split(data["eda"], labels,
                                                                                  test_size=ratio, random_state=seed)
        tempTrainData, tempTestData, tempTrainLabel, tempTestLabel = train_test_split(data["temp"], labels,
                                                                                  test_size=ratio, random_state=seed)
        hrTrainData, hrTestData, hrTrainLabel, hrTestLabel = train_test_split(data["hr"], labels,
                                                                                  test_size=ratio, random_state=seed)

        # Check that all labels are the same
        assert accTrainLabel.all() == edaTrainLabel.all() == tempTrainLabel.all() == hrTrainLabel.all()
        assert accTestLabel.all() == edaTestLabel.all() == tempTestLabel.all() == hrTestLabel.all()

        trainData = dict(acc=accTrainData, eda=edaTrainData, temp=tempTrainData, hr=hrTrainData)
        trainLabel = accTrainLabel
        testData = dict(acc=accTestData, eda=edaTestData, temp=tempTestData, hr=hrTestData)
        testLabel = accTestLabel

        return trainData, trainLabel, testData, testLabel

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

            # Third channel is added in order not to crush with Conv1D layer from TensorFlow
            hr_x = hr_x.reshape(-1, hr_x.shape[1], 1)
            eda_x = eda_x.reshape(-1, eda_x.shape[1], 1)
            temp_x = temp_x.reshape(-1, temp_x.shape[1], 1)

            # TODO: I am assuming that all vectors have same lenght. In case two consecutive EMAs are too close, the
            #  lenght of the vector can be shorter. It should be done something about this: either pad the vector or
            #  discard the vector. ATTENTION!

            acc_ = acc_x if len(acc_) == 0 else np.concatenate((acc_, acc_x), axis=0)
            hr_ = hr_x if len(hr_) == 0 else np.concatenate((hr_, hr_x), axis=0)
            eda_ = eda_x if len(eda_) == 0 else np.concatenate((eda_, eda_x), axis=0)
            temp_ = temp_x if len(temp_) == 0 else np.concatenate((temp_, temp_x), axis=0)


        # It is converted to float32, since it is default datatype for Tensorflow
        data_ = dict(acc=np.float32(acc_),
                     eda=np.float32(eda_),
                     hr=np.float32(hr_),
                     temp=np.float32(temp_))

        # Clean the less representative emotions captured by the smartwatch
        data_, label_ = self.__cleanDataset(data_, label_)

        if self.one_hot:
            label_ = self.__oneHotLabels(label_.reshape(-1,))

        if self.leave_one_out:
            return -1
        else:
            trainData, trainLabel, testData, testLabel = self.__splitTrainTest(data_, label_)

            #trainDataset = tf.data.Dataset.from_tensor_slices((trainData, trainLabel))
            #testDataset = tf.data.Dataset.from_tensor_slices((testData, testLabel))

            return trainData, trainLabel, testData, testLabel
