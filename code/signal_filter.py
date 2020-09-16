import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly as py
import plotly.graph_objs as go
import plotly.io as pio
from scipy import signal
from utils import getDataframe as gd
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt

def butter_bandpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')
    return b, a

def butter_bandpass_filter(data, lowcut, fs, order=5):
    b, a = butter_bandpass(lowcut, fs, order=order)
    y = lfilter(b, a, data, axis=0)
    return y


if __name__ == '__main__':

    path_to_eda = "data/datos_E4/P1/P1-530412_Complete/EDA3.csv"
    eda_col_name = ['EDA']
    eda_df = gd.get_dataframe(
        signal='eda',
        file_path=path_to_eda,
        column_names=eda_col_name)

    t = eda_df['index'][0:48]   # time axis
    x = eda_df['EDA'][0:48]
    #plt.plot(t, x)
    #plt.show()

    fs = 4  # sampling frequency
    fc = 1.5    # cut-off frequency


    y = butter_bandpass_filter(x, fc, fs, order=3)
    print(np.sum(y))
    plt.plot(t, y)
    plt.show()

    '''
    fs = 4
    plt.plot(eda_df['index']/fs, eda_df['EDA'], label='original')

    fc = 1.5
    w = fc / (fs / 2)
    b, a = signal.butter(3, w, 'low', fs=fs)
    output = signal.filtfilt(b, a, eda_df['EDA'])
    plt.plot(eda_df['index']/fs, output, label='filtered')
    plt.legend()
    plt.show()
    '''