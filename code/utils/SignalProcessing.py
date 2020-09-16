#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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
    with the information needed to feed the deep neural network. It will separate values according
    to a time window which size can modify when creating a class object instance.
    """

    type_signal = None
    col_name = None
    file_path = None

    def __init__(self, type_signal, path_to_file, path_to_ema, window_size, type_label):
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

    def __procEda(self):
        """
        This private method corresponds to the pre-processing step of the EDA signal
        :return: Adds the corresponding columns to the object data frame
        """
        self.df['eda_f'] = self.__filterSignal(data=self.df['eda'], type_filter='low')
        # Need to extract first 10 minutes of each signal!!!!

    def __procTemp(self):
        # Need to extract first 10 minutes of each signal!!!!
        pass

    def __procHr(self):
        # Need to extract first 10 minutes of each signal!!!!
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
        vectors = []
        labels = []

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

                # check if the vector size is the correct one, if not pad it
                if vector.shape[0] != self.fs * self.ws + 1:
                    pad_len = int(self.fs * self.ws - vector.shape[0])
                    npad = ((0, pad_len), (0, 0), (0, 0))
                    vector = np.pad(vector, pad_width=npad, mode='constant', constant_values=0)

            else:
                vector = df_aux[self.type_signal].to_numpy().reshape(-1, 1)  # Watch out the ACC signal

                if vector.shape[0] != self.fs * self.ws + 1:
                    pad_len = int(self.fs * self.ws - vector.shape[0])
                    npad = ((0, pad_len), (0, 0))  # Second element is to avoid to pad in the axis=1
                    vector = np.pad(vector, pad_width=npad, mode='constant', constant_values=0)

            # Label is the mode between l_bound and r_bound
            label = stat.mode(df_aux[self.label])

            vectors.append(vector)
            labels.append(label)

        vectors = np.array(vectors)
        labels = np.array(labels).reshape(-1, 1)  # Return a column vector

        return vectors, labels  # Must return something










    def getGroundTruth_v0(self):
        """
        This method takes the data frame computed in procSignal() method and chunk it according to the window
        size input.
        :return: A data frame with the discretized signal according to the input window size.
        """
        self.__procSignal()

        signal_chunks = []
        signal_chunks_array = []
        num_ema = len(self.ema_ts) - 1  # Since in first index in Series is 0
        for idx, ts in enumerate(self.ema_ts):   # ts stands for current timestamp in the for-loop
            # It is divided in three part since extreme timestamps, that is, the first and last timestamp
            # are analysed in a different fashion than mid-timestamps
            if idx == num_ema:      # Last EMA data points selection according to window size
                prev_ts = self.ema_ts[idx - 1]
                diff = ts - prev_ts
                available_data = len(self.df.loc[self.df['timestamp'] > ts])
                if diff > self.ws/2:  # Left EMA points selection of last EMA
                    #left_data = self.df[self.df['timestamp'].between(ts-self.ws/2, ts)]
                    left_data = self.df[self.df['timestamp'].between(ts-self.ws/2+self.Ts, ts)]
                else:
                    left_data = self.df[self.df['timestamp'].between(ts-(diff/2)+self.Ts, ts)]
                if available_data > self.ws/2:  # Right EMA points selection of first EMA
                    right_data = self.df[self.df['timestamp'].between(ts+self.Ts, ts+self.ws/2)]  # +Ts to avoid duplicates
                else:
                    right_data = self.df[self.df['timestamp'].between(ts+self.Ts, ts+available_data)]  # +Ts to avoid duplicates

            else:
                next_ts = self.ema_ts[idx+1]
                diff_next = next_ts - ts
                if idx == 0:    # First EMA data points selection according window size
                    available_data = len(self.df.loc[self.df['timestamp'] < ts])
                    if available_data > self.ws/2:    # Left EMA points selection of first EMA
                        #left_data = self.df[self.df['timestamp'].between(ts-self.ws/2, ts)]
                        left_data = self.df[self.df['timestamp'].between(ts-self.ws/2+self.Ts, ts)]
                    else:
                        #left_data = self.df[self.df['timestamp'].between(ts-available_data, ts)]
                        left_data = self.df[self.df['timestamp'].between(ts-available_data+self.Ts, ts)]
                    if diff_next > self.ws/2:       # Right EMA points selection of first EMA
                        right_data = self.df[self.df['timestamp'].between(ts+self.Ts, ts+(self.ws/2))]

                    else:
                        right_data = self.df[self.df['timestamp'].between(ts+self.Ts, ts+diff_next/2-self.Ts)]


                else:       # Middle EMA data points selection according window size
                    prev_ts = self.ema_ts[idx-1]
                    diff_prev = ts - prev_ts
                    if diff_prev > self.ws/2:       # Left EMA points selection of mid EMA
                        #left_data = self.df[self.df['timestamp'].between(ts-self.ws/2, ts)]
                        left_data = self.df[self.df['timestamp'].between(ts-self.ws/2+self.Ts, ts)]
                    else:
                        #left_data = self.df[self.df['timestamp'].between(ts-diff_prev/2, ts)]
                        left_data = self.df[self.df['timestamp'].between(ts-diff_prev/2+self.Ts, ts)]
                    if diff_next > self.ws/2:       # Right EMA points selection of mid EMA
                        right_data = self.df[self.df['timestamp'].between(ts+self.Ts, ts+self.ws/2)]

                    else:
                        right_data = self.df[self.df['timestamp'].between(ts+self.Ts, ts+diff_next/2)]

            left_right_data = pd.concat([left_data, right_data])
            #left_right_array = left_right_data[self.type_signal].to_numpy().reshape(1, -1)
            # Append timestamp block with its corresponding window size
            signal_chunks.append(left_right_data)
            #signal_chunks_array.append(left_right_array)

        array_final = np.array(signal_chunks_array)
        df_final = pd.concat(signal_chunks)
        return df_final

    def getGroundTruth_v1(self):
        """
        This method is a modified version of the previous method
        In this case, the first and last timestamps are just ignored
        :return:
        """
        self.__procSignal()
        ema_ts = self.ema_ts[1:-1].reset_index(drop=True)   # Get rid of the first and last value: they are references
        num_ema = len(ema_ts) - 1  # "-1" since in first index in Series is 0
        signal_chunks = []
        signal_chunks_array = []
        for idx, ts in enumerate(ema_ts):
            if idx == num_ema:  # Last EMA data points selection according to window size
                prev_ts = ema_ts[idx - 1]
                diff = ts - prev_ts     # code could be reduce by using np.abs of the difference
                available_data = len(self.df.loc[self.df['timestamp'] > ts])
                if diff > self.ws / 2:  # Left EMA points selection of last EMA
                    # left_data = self.df[self.df['timestamp'].between(ts-self.ws/2, ts)]
                    left_data = self.df[self.df['timestamp'].between(ts - self.ws / 2 + self.Ts, ts)]
                else:
                    left_data = self.df[self.df['timestamp'].between(ts - (diff / 2) + self.Ts, ts)]
                if available_data > self.ws / 2:  # Right EMA points selection of first EMA
                    right_data = self.df[self.df['timestamp'].between(ts + self.Ts, ts + self.ws / 2)]  # +Ts to avoid duplicates
                else:
                    right_data = self.df[self.df['timestamp'].between(ts + self.Ts, ts + available_data)]  # +Ts to avoid duplicates

            else:
                next_ts = ema_ts[idx + 1]
                diff_next = next_ts - ts
                if idx == 0:  # First EMA data points selection according window size
                    available_data = len(self.df.loc[self.df['timestamp'] < ts])
                    if available_data > self.ws / 2:  # Left EMA points selection of first EMA
                        # left_data = self.df[self.df['timestamp'].between(ts-self.ws/2, ts)]
                        left_data = self.df[self.df['timestamp'].between(ts - self.ws / 2 + self.Ts, ts)]
                    else:
                        # left_data = self.df[self.df['timestamp'].between(ts-available_data, ts)]
                        left_data = self.df[self.df['timestamp'].between(ts - available_data + self.Ts, ts)]
                    if diff_next > self.ws / 2:  # Right EMA points selection of first EMA
                        right_data = self.df[self.df['timestamp'].between(ts + self.Ts, ts + (self.ws / 2))]
                    else:
                        right_data = self.df[self.df['timestamp'].between(ts + self.Ts, ts + diff_next / 2 - self.Ts)]

                else:  # Middle EMA data points selection according window size
                    prev_ts = ema_ts[idx - 1]
                    diff_prev = ts - prev_ts
                    if diff_prev > self.ws / 2:  # Left EMA points selection of mid EMA
                        # left_data = self.df[self.df['timestamp'].between(ts-self.ws/2, ts)]
                        left_data = self.df[self.df['timestamp'].between(ts - self.ws / 2 + self.Ts, ts)]
                        right_data = self.df[self.df['timestamp'].between(ts + self.Ts, ts + self.ws / 2)]
                    else:
                        # left_data = self.df[self.df['timestamp'].between(ts-diff_prev/2, ts)]
                        left_data = self.df[self.df['timestamp'].between(ts - diff_prev / 2 + self.Ts, ts)]
                        right_data = self.df[self.df['timestamp'].between(ts + self.Ts, ts + diff_next / 2)]


            left_right_data = pd.concat([left_data, right_data])
            if self.type_signal != 'acc':
                vector = left_right_data[self.type_signal].to_numpy()
            else:
                # Instead of forming an array here, we just keep the vectors in case we need to pad them. Later it will
                # be formed the 4-dimensional array
                vector_xf = left_right_data['x_f'].to_numpy().reshape(1, -1)
                vector_yf = left_right_data['y_f'].to_numpy().reshape(1, -1)
                vector_zf = left_right_data['z_f'].to_numpy().reshape(1, -1)
                vector_nf = left_right_data['n_f'].to_numpy().reshape(1, -1)


            '''
            This is a pad module. This must be kept until we know what to do with those variable vectors.
            To form the dataset in TensorFlow all vector must have same length.
            '''
            if self.type_signal != 'acc':
                if len(vector) == self.ws * self.fs:
                    pass    # No need to pad
                else:
                    dif = self.ws * self.fs - len(vector)
                    if (dif % 2) == 0:
                        pad_width = int(dif / 2)
                        #vector_pad = np.pad(np.squeeze(vector), (0, 2 * pad_width), mode='mean')
                        vector_pad = np.pad(np.squeeze(vector), (0, 2 * pad_width), mode='constant',
                                            constant_values=(0, 0))

                    else:
                        pad_width = int(dif / 2)
                        #vector_pad = np.pad(np.squeeze(vector), (0, 2 * pad_width + 1), mode='mean')
                        vector_pad = np.pad(np.squeeze(vector), (0, 2 * pad_width + 1), mode='constant',
                                            constant_values=(0, 0))

                    assert len(vector_pad) == self.ws * self.fs , "Signal vector does not have correct length!"
                    vector = vector_pad
            else:
                if vector_xf.shape[-1] == self.ws * self.fs and vector_yf.shape[-1] == self.ws * self.fs \
                        and vector_zf.shape[-1] == self.ws * self.fs and vector_nf.shape[-1]: # if all ACC vector have correct length
                    # we have to build up here vector variable not to break code
                    vector = np.zeros([4, 1, vector_xf.shape[-1]])
                    vector[0:1:] = vector_xf
                    vector[1:1:] = vector_yf
                    vector[2:1:] = vector_zf
                    vector[3:1:] = vector_nf

                else:
                    dif_xf = self.ws * self.fs - len(vector_xf) # we assume we have same number of point for 4 columns
                    if (dif_xf % 2) == 0:
                        pad_width = int(dif_xf / 2)

                        vector_xf_pad = np.pad(np.squeeze(vector_xf), (0, 2 * pad_width), mode='constant', constant_values=(0 ,0))
                        vector_yf_pad = np.pad(np.squeeze(vector_yf), (0, 2 * pad_width), mode='constant', constant_values=(0 ,0))
                        vector_zf_pad = np.pad(np.squeeze(vector_zf), (0, 2 * pad_width), mode='constant', constant_values=(0 ,0))
                        vector_nf_pad = np.pad(np.squeeze(vector_nf), (0, 2 * pad_width), mode='constant', constant_values=(0 ,0))


                    else:
                        pad_width = int(dif_xf / 2)

                        vector_xf_pad = np.pad(np.squeeze(vector_xf), (0, 2 * pad_width + 1), mode='constant', constant_values=(0 ,0))
                        vector_yf_pad = np.pad(np.squeeze(vector_yf), (0, 2 * pad_width + 1), mode='constant', constant_values=(0 ,0))
                        vector_zf_pad = np.pad(np.squeeze(vector_zf), (0, 2 * pad_width + 1), mode='constant', constant_values=(0 ,0))
                        vector_nf_pad = np.pad(np.squeeze(vector_nf), (0, 2 * pad_width + 1), mode='constant', constant_values=(0 ,0))



                    assert len(vector_xf_pad) == self.ws * self.fs and len(vector_yf_pad) == self.ws * self.fs \
                           and len(vector_zf_pad) == self.ws * self.fs and len(vector_nf_pad) , 'ACC vector signals must have' \
                                                                                        'same length!'
                    # create vector --> In this case it will be a 3-dimensional array

            signal_chunks_array.append(vector)

        array_vector = np.array(signal_chunks_array)
        if self.label == 'happines':
            array_label = self.happiness[1:-1]
        elif self.label == 'arousal':
            array_label = self.arousal[1:-1]
        elif self.label == 'mood':
            array_label = self.mood[1:-1]
        else:
            array_label = None

        return array_vector, array_label.to_numpy().reshape(-1, 1)

