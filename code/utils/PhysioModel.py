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
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, MaxPooling1D


class PhysioModel(Model):
    def __init__(self, num_classes, batch_size):
        super(PhysioModel, self).__init__()
        # Declare all layers in raw

        self.conv1 = Conv1D(filters=32, kernel_size=16, activation='relu', input_shape=(None, 1920, 4),
                                dtype='float32', name='conv1_ACC')
        self.conv2 = Conv1D(filters=32, kernel_size=16, activation='relu', dtype='float32', name='conv2_ACC')
        self.dropOut = Dropout(0.5)
        self.pool = MaxPooling1D(pool_size=2, name='poolACC')
        self.flatten = Flatten()
        self.fc = Dense(100, activation='relu', dtype='float32', name='fc')
        self.logits = Dense(num_classes, activation='softmax', dtype='float32', name='logits')


        '''
        self.conv1_acc = Conv1D(32, 16, activation='relu', input_shape=(batch_size, 1920, 3),
                                dtype='float32', name='convACC')
        self.conv1_eda = Conv1D(32, 16, activation='relu', input_shape=(batch_size, 240, 1),
                                dtype='float32', name='convEDA')
        self.conv1_hr = Conv1D(32, 8, activation='relu', input_shape=(batch_size, 60, 1),
                               dtype='float32', name='convHR')
        self.conv1_temp = Conv1D(32, 16, activation='relu', input_shape=(batch_size, 240, 1),
                                 dtype='float32', name='convTEMP')
        self.flatten = Flatten()
        self.fc = Dense(64, activation='relu', dtype='float32', name='fc')
        self.logits = Dense(out_label, dtype='float32', name='logits')
        '''
    # We assume x will be "data" variable

    def call(self, inputs):

        x_in = inputs["acc"]
        #print("La forma que tiene el vector conv es: {}".format(x_in.shape))
        # print("La forma que tiene el vector conv es: {}".format(x.shape))
        # print("La forma que tiene el vector flatten es: {}".format(x.shape))
        # print("La forma que tiene el vector fc es: {}".format(x.shape))
        # print("La forma que tiene el vector logits es: {}".format(x.shape))


        x = self.conv1(x_in)
        x = self.conv2(x)
        x = self.dropOut(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.logits(x)


        return x




        '''
        with tf.name_scope("ACC") as scope:
            x_acc = self.conv1_acc(inputs["acc"])
            x_acc_flat = self.flatten(x_acc)

        with tf.name_scope("HR") as scope:
            x_hr = self.conv1_hr(inputs["hr"])
            x_hr_flat = self.flatten(x_hr)

        with tf.name_scope("EDA") as scope:
            x_eda = self.conv1_eda(inputs["eda"])
            x_eda_flat = self.flatten(x_eda)
        with tf.name_scope("TEMP") as scope:
            x_temp = self.conv1_temp(inputs["temp"])
            x_temp_flat = self.flatten(x_temp)

        x = concatenate([x_acc_flat, x_hr_flat, x_eda_flat, x_temp_flat])
        x = self.fc(x)

        logits = self.logits(x)
        # record summaries
        tf.summary.histogram('outputs', logits)

        return logits
        '''





    '''
    def model(self):
        x_acc = tf.keras.Input(shape=(1920, 4))
        x_hr = tf.keras.Input(shape=(60, 1))
        x_eda = tf.keras.Input(shape=(240, 1))
        x_temp = tf.keras.Input(shape=(240, 1))
    '''


        #x = self.flatten(concatenate([x_acc, x_hr, x_eda, x_temp]))
        #x = self.fc(x)
        #return self.logits(x)