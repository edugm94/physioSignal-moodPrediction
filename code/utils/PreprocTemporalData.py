#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
#   Author: Eduardo Gutierrez Maestro
#   Date: 2021.02.08
#   email: eduardo.gutierrez-maestro@oru.se
#
#   Center for Applied Autonomous Sensor Systems (AASS), Cognitive Robotic Systems Labs
#   University of Orebro, Sweden
#

from sklearn.preprocessing import MinMaxScaler
import numpy as np

class PreprocTemporalData:
    def __init__(self, trainDataset, testDataset):
        self.train_ds = trainDataset
        self.test_ds = testDataset

    def __prepocACC(self):
        range=(-1, 1)
        train_acc = self.train_ds['acc']
        test_acc = self.test_ds['acc']

        train_x = train_acc[:, :, 0]
        train_y = train_acc[:, :, 1]
        train_z = train_acc[:, :, 2]
        train_n = train_acc[:, :, 3]

        scaler_x = MinMaxScaler(feature_range=range)
        scaler_y = MinMaxScaler(feature_range=range)
        scaler_z = MinMaxScaler(feature_range=range)
        scaler_n = MinMaxScaler(feature_range=range)

        scaler_x.fit(train_x)
        scaler_y.fit(train_y)
        scaler_z.fit(train_z)
        scaler_n.fit(train_n)

        # Transformation of the training data
        train_x_norm = scaler_x.transform(train_x)
        train_y_norm = scaler_y.transform(train_y)
        train_z_norm = scaler_z.transform(train_z)
        train_n_norm = scaler_n.transform(train_n)

        # Transformation of the test data
        test_x_norm = scaler_x.transform(test_acc[:, :, 0])
        test_y_norm = scaler_y.transform(test_acc[:, :, 1])
        test_z_norm = scaler_z.transform(test_acc[:, :, 2])
        test_n_norm = scaler_n.transform(test_acc[:, :, 3])

        train_norm = np.dstack((train_x_norm, train_y_norm, train_z_norm, train_n_norm))
        test_norm = np.dstack((test_x_norm, test_y_norm, test_z_norm, test_n_norm))

        self.train_ds['acc'] = train_norm
        self.test_ds['acc'] = test_norm


    def __call__(self, trainDataset, testDataset):
        '''Here it should be call methods that will normalize data to input Nerual Network'''
        self.__prepocACC()

        return self.train_ds, self.test_ds






def preprocData(trainData, testData):
    def preprocAccSignal(trainACC, testACC):
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler.fit(trainACC)
        trainACC_norm = scaler.transform(trainACC)
        testACC_norm = scaler.transform(testACC)

        return trainACC_norm, testACC_norm

    trainACC, testACC = preprocAccSignal(trainData['acc'], testData['acc'])
    trainData['acc'] = trainACC
    testData['acc'] = testACC

    return trainData, testData

