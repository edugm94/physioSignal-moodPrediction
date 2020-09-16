from datetime import datetime
from datetime import timedelta
import pandas as pd

def get_dataframe(signal, file_path, column_names):
    def get_sample_time(index, init_time, Ts):
        aux = []
        for i in index:
            aux.append((init_time + timedelta(seconds=Ts * i)).strftime("%H:%M:%S:%f"))

        return aux

    df = pd.read_csv(file_path, names=column_names, header=None)
    timestamp = df.iloc[0, :][0]

    if signal in ('acc', 'eda', 'hr', 'temp', 'bvp'):
        initial_time = datetime.utcfromtimestamp(timestamp)
        fs = df.iloc[1, :][0]
        Ts = 1 / fs
        df = df.drop([0, 1]).reset_index(drop=True).reset_index()
        df.loc[:, 'time'] = get_sample_time(df['index'], initial_time, Ts)
    else:
        df = df.drop([0, 1]).reset_index(drop=True).reset_index()
        df['time'] = df.apply(lambda x:
                              (datetime.utcfromtimestamp(x['diff'] + timestamp).strftime("%H:%M:%S:%f")), axis=1)
    return df