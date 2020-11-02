#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
#   Author: Eduardo Gutierrez Maestro
#   Date: 2020.09.22
#   email: eduardo.gutierrez-maestro@oru.se
#
#   Center for Applied Autonomous Sensor Systems (AASS), Cognitive Robotic Systems Labs
#   University of Orebro, Sweden
#

from scipy import stats
import numpy as np


class SlidingWindow:
    def __init__(self, vectors, labels, overlapping, samp_freq, window_feat_size, signal_type):
        self.ov = overlapping
        self.fs = samp_freq
        self.wf = window_feat_size
        self.signal_type = signal_type
        self.vectors = vectors
        self.labels = labels

        # Check if vector and label vector's dimension is the correct one
        if vectors.shape[1] != labels.shape[1]:
            raise NameError('Dimension error!\n'
                            'Vectors dimension does not match with labels dimension. Check input please.')


    def extractRawVector(self):

        #ini = (w * (st - 1) + 10 + a1 - (ov / 100) * (st - 1) * w) * f + 1
        #fin = (w * st + 10 + a1 - (ov / 100) * (st - 1) * w) * f;

        st = 1
        cont = self.vectors.shape[1]
        num = 0
        vectors_matrix = []
        while cont >= (self.wf*self.fs):

            ini = int((self.wf * (st-1) - (self.ov/100)*(st-1)*self.wf)*self.fs)
            fin = int((self.wf * st - (self.ov/100) * (st-1) * self.wf)*self.fs)



            if self.signal_type != 'acc':

                #   Feature vector with an overlapping set by user
                raw_vector = self.vectors[:, ini:fin]
                raw_label = self.labels[:, ini:fin]

                # In order to compute actual label of the vector, compute the statistical mode
                actual_label = stats.mode(raw_label[0])[0].reshape(1, 1)
                # It is reshaped for the following concatenation

                vector_with_label = np.concatenate((actual_label, raw_vector), axis=1)    # Transform to column vector
                vectors_matrix.append(vector_with_label)                       # wont be needed when data is transformed
            else:
                # Since the ACC signal has 4 channels, the raw vector is done in a different way
                raw_vector = self.vectors[:, ini:fin, :]
                raw_label = self.labels[:, ini:fin]

                # In order to compute actual label of the vector, compute the statistical mode
                actual_label = stats.mode(raw_label[0])[0].reshape(1, 1)

                actual_label = np.tile(actual_label, (1, 1, 4))
                vector_with_label = np.concatenate((actual_label, raw_vector), axis=1)
                vectors_matrix.append(vector_with_label)

            st += 1
            num += 1
            cont = self.vectors.shape[1] - fin + (self.ov / 100) * self.wf

            #                           !!!!!!!!  OJO !!!!!!!!!!!
            # Es necesario decidir que hacer con los datos que sobran: es decir los datos del final de la señal que
            # no llegan a formar una ventana completa, son saltados por el bucle while.
            # OJO: Hay que pensar si hacer un vector mas relleno de ceros o ignorar esos datos.

        vectors_matrix = np.vstack(vectors_matrix)
        return vectors_matrix
