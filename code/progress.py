import time
import progressbar
from tqdm import tqdm, trange
from time import sleep
import sys


'''
bar_u = progressbar.ProgressBar(max_value=2, prefix='User: ')
bar_d = progressbar.ProgressBar(max_value=20, prefix='User: ')
for i in range(2):
    for j in range(20):
        time.sleep(0.1)
        bar_d.update(j)
    bar_u.update(i)
'''
'''
for i in tqdm(range(5), desc='1st loop', position=0):
    for j in tqdm(range(100), desc='2nd loop', position=1):
        sleep(0.01)

'''


from tqdm import tqdm, trange
from random import random, randint
from time import sleep
'''
with trange(10) as t:
    for i in t:
        # Description will be displayed on the left
        t.set_description('GEN %i' % i)
        # Postfix will be displayed on the right,
        # formatted automatically based on argument's datatype
        t.set_postfix(loss=random(), gen=randint(1,999), str='h',
                      lst=[1, 2])
        sleep(0.1)


with tqdm(total=10, bar_format="{postfix[0]} {postfix[1][value]:>8.2g}",
          postfix=["Batch", dict(value=0)]) as t:
    for i in range(10):
        sleep(0.1)
        t.postfix[1]["value"] = i / 2
        t.update()
'''

for i in range(4):
    with tqdm(range(100), position=0) as t_d:
        for j in t_d:
            t_d.set_description('Patient {} | day {}: '.format(i+1, j+1))
            sleep(0.1)
