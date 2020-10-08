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
import pandas as pd
import os
import numpy as np
from scipy.signal import butter, lfilter
from scipy.interpolate import interp1d
from datetime import datetime
import statistics as stat


class SignalProcessing:
    """
    This class aim is to process the input physiological signals, creating a Dataframe in Pandas
    with the information needed to feed the deep neural network.
    The returned data is named as significant data, i.e data which falls within the set window size
    around each answered EMA.
    It will be returned both: vectors of filtered signals, and expanded labels.
    """

    type_signal = None
    col_name = None
    file_path = None

    def __init__(self, type_signal, path_to_file, path_to_ema, window_size, type_label, all):
        if not os.path.exists(path_to_file):
            raise NameError('Wrong path!\n'
                            'Input a existing data file path, please.')
        if not os.path.exists(path_to_ema):
            raise NameError('Wrong path!\n'
                            'Input a existing EMA path, please.')
        if type_label.lower() not in ('happiness', 'arousal', 'mood'):
            raise NameError('Wrong label name!\n'
                            'Input a correct EMA label (e.g: "happiness", "arousal", "mood"), please.')
        self.type_signal = type_signal.lower()
        self.file_path = path_to_file
        self.ema_path = path_to_ema
        self.col_name = None
        self.df = None
        self.fs = None
        self.Ts = None
        self.init_ts = None
        self.ema_ts = None
        self.ws = window_size * 60  # Window size input in minutes time unit -->  Convert to seconds
        self.happiness = None   # ¡OJO!: Could be deleted --> it is only necessary label attribute
        self.arousal = None     # ¡OJO!: Could be deleted --> it is only necessary label attribute
        self.mood = None        # ¡OJO!: Could be deleted --> it is only necessary label attribute
        self.label = type_label.lower() # can be 'happiness', 'arousal' or 'mood'
        self.all = all          # Attribute which indicates if it will be used all patients data or just one patient
                                # (Needed to know in order to normalize signals)
    def __readCSV(self):
        if self.type_signal in ('acc', 'eda', 'hr', 'temp', 'bvp'):
            self.df = pd.read_csv(self.file_path, names=self.col_name, header=None)
            self.fs = self.df.iloc[1, :][0]
            self.Ts = 1 / self.fs
            #   self.init_ts = self.df.iloc[0, :][0]
            self.init_ts = self.df.iloc[0, :][0] if self.type_signal == 'hr' else self.df.iloc[0, :][0] + 10
            # ¡¡¡ WATCH OUT YOU ARE ADDING +10 TO TIMESTAMPS !!! --> THIS IS DONE TO MAKE THEM HAVE SAME STARTING TIME
            self.df = self.df.drop([0, 1]).reset_index(drop=True).reset_index()
            self.df['timestamp'] = self.init_ts + self.df['index']*self.Ts
            self.df['time'] = self.df['timestamp'].apply(datetime.fromtimestamp)

        else:   # Not reading currently IBI signals, since they are not used in the original paper
            pass

    def __butter_lowpass(self, lowcut=1.5, order=3):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        b, a = butter(order, low, btype='low')
        return b, a

    def __butter_bandpass(self, lowcut=0.2, highcut=10, order=3):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def __butter_highpass(self, highcut=10., order=3):
        nyq = 0.5 * self.fs
        high = highcut / nyq
        b, a = butter(order, high, btype='high')
        return b, a

    def __filterSignal(self, data, type_filter):
        if type_filter == 'low':
            b, a = self.__butter_lowpass()
            return lfilter(b, a, data, axis=0)
        elif type_filter == 'band':
            b, a = self.__butter_bandpass()
            return lfilter(b, a, data, axis=0)
        elif type_filter == 'high':
            b, a = self.__butter_highpass()
            return lfilter(b, a, data, axis=0)

    def __computeNorm(self):
        """
        This private method is created to compute the norm of the whole .CSV file using the built-in
        fucntions provided by numpy. With df.apply() functionality, a for-loop was created increasing
        considerable the execution time.

        :return: An array of the with the norm of the 3 variables of the accelerometer: x, y, z.
        """
        # First it is obtained 3 variables from the dataframe
        x = self.df['x'].to_numpy()
        y = self.df['y'].to_numpy()
        z = self.df['z'].to_numpy()

        # Need to be reshape in order to concatenate them
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        z = z.reshape(z.shape[0], -1)

        # It is formed an unified array to compute directly
        aux = np.concatenate((x, y, z), axis=1)
        return np.linalg.norm(aux, axis=1)

    def __procAcc(self):
        """
        This private method corresponds to the pre-processing step of the accelerometer signal.
        :return: Adds the corresponding columns to the object data frame.
        """

        self.df['n'] = self.__computeNorm()
        self.df['x_f'] = self.__filterSignal(data=self.df['x'], type_filter='band')
        self.df['y_f'] = self.__filterSignal(data=self.df['y'], type_filter='band')
        self.df['z_f'] = self.__filterSignal(data=self.df['z'], type_filter='band')
        self.df['n_f'] = self.__filterSignal(data=self.df['n'], type_filter='band')

        # Need to extract first 10 minutes of each signal!!!!
        if self.all == 1:
            pass


    def __procEda(self):
        """
        This private method corresponds to the pre-processing step of the EDA signal
        :return: Adds the corresponding columns to the object data frame
        """
        self.df['eda_f'] = self.__filterSignal(data=self.df['eda'], type_filter='low')

        # Need to extract first 10 minutes of each signal!!!!
        if self.all == 1:   # Need to be extract first 10 min
            pass


    def __procTemp(self):
        # Need to extract first 10 minutes of each signal!!!!
        if self.all == 1:
            pass
        pass

    def __procHr(self):
        # Need to extract first 10 minutes of each signal!!!!
        if self.all == 1:
            pass
        pass

    def __assignLabel(self):
        """
        This method assign each data point a label of the 3 categories evaluated: happiness; arousal; mood.
        Each label is assigned by the interpolation with the timestamp when the EMA was answered.

        :return: Adds a column in the final dataframe:
                df['y_ha'] --> happiness label
                df['y_ar'] --> arousal label
                df['y_mo'] --> mood label
        """
        label_df = pd.read_excel(self.ema_path)
        self.happiness = label_df.iloc[:, 5][1:-1]  # Drop first and last value from EMA excel column (for all values)
        self.arousal = label_df.iloc[:, 6][1:-1]
        self.mood = label_df.iloc[:, 7][1:-1]
        ema_timestamp_aux = label_df.iloc[:, 4]
        self.end_ts = self.init_ts + ema_timestamp_aux.iloc[-1]   # We keep last 'ts' (time where stop collecting data)
        ema_timestamp_aux = ema_timestamp_aux[1:-1]
        self.ema_ts = self.init_ts + np.squeeze(ema_timestamp_aux.to_frame().apply(np.ceil)) #contains actual 'ts'

        ################  Test module  ##################
        #This block just prove the computation to obtain the timestamp values for the different signals
        #aux_df = self.ema_ts.to_frame()
        #aux_df['time_ema'] = aux_df['Interp (s)'].apply(datetime.fromtimestamp)
        #################################################

        ha_interp = interp1d(self.ema_ts, self.happiness, kind='nearest', fill_value="extrapolate")
        ar_interp = interp1d(self.ema_ts, self.arousal, kind='nearest', fill_value="extrapolate")
        mo_interp = interp1d(self.ema_ts, self.mood, kind='nearest', fill_value="extrapolate")

        aux_ha = ha_interp(self.df['timestamp'])
        aux_ar = ar_interp(self.df['timestamp'])
        aux_mo = mo_interp(self.df['timestamp'])

        self.df['happiness'] = aux_ha.astype(int)
        self.df['arousal'] = aux_ar.astype(int)
        self.df['mood'] = aux_mo.astype(int)

    def __procSignal(self):
        """
        This method will read each CSV file according to the signal type attribute. In case it is required, the
        signal will filter the signal.
        :return: It creates a DataFrame within the class object instance.
        """
        if self.type_signal == 'acc':
            self.col_name = ['x', 'y', 'z']
            self.__readCSV()
            self.__procAcc()
        elif self.type_signal == 'eda':
            self.col_name = ['eda']
            self.__readCSV()
            self.__procEda()
        elif self.type_signal == 'temp':
            self.col_name = ['temp']
            self.__readCSV()
            self.__procTemp()  # This function does nothing; just substract 10 first minutes if needed
        elif self.type_signal == 'hr':
            self.col_name = ['hr']
            self.__readCSV()
            self.__procHr()  # This function does nothing; just substract 10 first minutes if needed

        self.__assignLabel()        # Call private method to assign each data point a label

    def getGroundTruth(self):
        """
        This method gets the corresponding data vectors and label according to a fixed signal and window size.
        The main idea of this code is to obtain for each answered EMA the initial and final timestamps, so that, the
        equality among EMA in terms of datapoints splitting is held.

        :return: A pair vector-label arrays.
            @vectors: matrix containing the vectors for each answered EMA
            @labels: matrix/vector containing corresponding label for each vector
        """

        self.__procSignal()  # This function will read CSV file and filter the corresponding signal

        n_ts = len(self.ema_ts)  # Total number of EMA

        # Initializacion of the accumulative data array
        if self.type_signal == 'acc':
            vectors = np.array([]).reshape(-1, 1, 4)
        else:
            vectors = np.array([]).reshape(-1, 1)
        #array where all all significat labels will be stored
        labels = np.array([]).reshape(-1, 1)

        for idx, ts in enumerate(self.ema_ts):
            idx += 1  # It is added 1 because Pandas Series starts with 1 index

            if idx in (1, n_ts):
                if idx == 1:
                    l_dis = ts - self.init_ts  # Calculate num. of points between current ts and initial ts
                    if l_dis < self.ws / 2:
                        l_bound = self.init_ts  # This corresponds to the left-side timestamp
                    else:
                        l_bound = ts - self.ws / 2
                    r_dis = self.ema_ts[idx + 1] - ts  # Calculate num. of points between current ts and next ts
                    if r_dis < self.ws:  # Datapoints are distributed equally
                        r_bound = ts + r_dis / 2  # This corresponds to the right-side timestamp
                    else:
                        r_bound = ts + self.ws / 2
                elif idx == n_ts:  # This process is the same as the upper one but for the last timestamp
                    l_dis = ts - self.ema_ts[idx - 1]
                    if l_dis < self.ws:
                        l_bound = ts - l_dis / 2
                    else:
                        l_bound = ts - self.ws / 2
                    r_dis = self.end_ts - ts
                    if r_dis < self.ws / 2:
                        r_bound = self.end_ts
                    else:
                        r_bound = ts + self.ws / 2
            else:
                l_dis = ts - self.ema_ts[idx - 1]
                if l_dis < self.ws:
                    l_bound = ts - l_dis / 2
                else:
                    l_bound = ts - self.ws / 2
                r_dis = self.ema_ts[idx + 1] - ts
                if r_dis < self.ws:
                    r_bound = ts + r_dis / 2
                else:
                    r_bound = ts + self.ws / 2

            # The next code is equivalent to all previous steps
            df_aux = self.df[self.df['timestamp'].between(l_bound, r_bound)]

            # Vector is data of a specific timestamp
            if self.type_signal == 'acc':
                vector = np.zeros([df_aux.shape[0], 1, 4])
                vector[:, :, 0] = df_aux['x_f'].to_numpy().reshape(-1, 1)
                vector[:, :, 1] = df_aux['y_f'].to_numpy().reshape(-1, 1)
                vector[:, :, 2] = df_aux['z_f'].to_numpy().reshape(-1, 1)
                vector[:, :, 3] = df_aux['n_f'].to_numpy().reshape(-1, 1)
            else:
                vector = df_aux[self.type_signal].to_numpy().reshape(-1, 1)  # Watch out the ACC signal

            # Label is the mode between l_bound and r_bound
            label = df_aux[self.label].to_numpy().reshape(-1, 1)

            vectors = np.append(vectors, vector, axis=0)
            labels = np.append(labels, label, axis=0)

        vectors = np.array(vectors)
        labels = np.array(labels).reshape(-1, 1)  # Return a column vector

        return vectors, labels  # Must return something
