#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
#   Author: Eduardo Gutierrez Maestro
#   Date: 2021.01.14
#   email: eduardo.gutierrez-maestro@oru.se
#
#   Center for Applied Autonomous Sensor Systems (AASS), Cognitive Robotic Systems Labs
#   University of Orebro, Sweden
#

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Flatten, Dense, concatenate


class PhysioModel(Model):
    def __init__(self, out_label, batch_size):
        super(PhysioModel, self).__init__()
        # Declare all layers in raw
        self.conv1_acc = Conv1D(32, 2, activation='relu', input_shape=(batch_size, 1920, 3),
                                dtype='float64', name='convACC')
        self.conv1_eda = Conv1D(32, 2, activation='relu', input_shape=(batch_size, 240, 1),
                                dtype='float64', name='convEDA')
        self.conv1_hr = Conv1D(32, 2, activation='relu', input_shape=(batch_size, 60, 1),
                               dtype='float64', name='convHR')
        self.conv1_temp = Conv1D(32, 2, activation='relu', input_shape=(batch_size, 240, 1),
                                 dtype='float64', name='convTEMP')
        self.flatten = Flatten()
        self.fc = Dense(128, activation='relu', dtype='float64')
        self.logits = Dense(out_label)

    # We assume x will be "data" variable
    def call(self, inputs):

        with tf.name_scope("ACC") as scope:
            x_acc = self.conv1_acc(inputs["acc"])

        with tf.name_scope("HR") as scope:
            x_hr = self.conv1_hr(inputs["hr"])

        with tf.name_scope("EDA") as scope:
            x_eda = self.conv1_eda(inputs["eda"])

        with tf.name_scope("TEMP") as scope:
            x_temp = self.conv1_temp(inputs["temp"])

        x_acc_flat = self.flatten(x_acc)
        x_hr_flat = self.flatten(x_hr)
        x_eda_flat = self.flatten(x_eda)
        x_temp_flat = self.flatten(x_temp)

        x = concatenate([x_acc_flat, x_hr_flat, x_eda_flat, x_temp_flat])
        x = self.fc(x)
        return self.logits(x)


        #x = self.flatten(concatenate([x_acc, x_hr, x_eda, x_temp]))
        #x = self.fc(x)
        #return self.logits(x)