import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


'''
df = pd.DataFrame({
    'x' : [1, 2, 3],
    'y' : [4, 5, 6],
    'z' : [7, 8, 9]
})

def norm(x):
    return np.linalg.norm([x['x'], x['y']])

#df['n'] = df.apply(norm, axis=1)
#print(type(df['x']))
#df['n'] = np.linalg.norm([df['x'], df['y']])
#print(df.head())


x = np.array([[1, 2, 3]])
y = np.array([[4, 5, 6]])
z = np.array([[7, 8, 9]])

aux = np.concatenate((x.T,y.T, z.T), axis=1)
n = np.linalg.norm(aux, axis=1)

print(aux)
print(n)
'''
'''
path_ema = "data/datos_E4/P1/P1-530412_Complete/EMAs1.xlsx"
df = pd.read_excel(path_ema)
l = np.linspace(1, 100, 100, dtype=np.int)
mood = df.iloc[:, 7]
timestamp = df.iloc[:, 4]

moo = interp1d(timestamp, mood)
aux = moo(l).reshape(1, -1)
aux = aux.astype(int)
print(len(moo(l)))
'''

x = np.linspace(10, 22, num=10)
y = np.sin(x)

f = interp1d(x, y)
xx = np.linspace(10, 20, num=100)
yy = f(xx)