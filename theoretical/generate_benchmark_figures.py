'''
Generate plots for the benchmark evaluation.
'''

import numpy as np
import pandas as pd
import seaborn as sns

from glob import glob
import os

import matplotlib.pyplot as plt

files = glob('Theory/results/*.csv')

dfs = []

for f in files:
    df = pd.read_csv(f)
    df.loc[df.index[-1],'Model'] = 'Rank_1'
    df.loc[df.index[-2],'Model'] = 'Local'
    dfs.append(df)

df = pd.concat(dfs,ignore_index=True,axis=0)

fig=plt.figure()
fig.set_size_inches(8,6)
sns.stripplot(x='Patches',y='time',data=df[df['Model']=='Patch'])
plt.yscale('log')
plt.xlabel('Number of patches per dimension',fontsize=20)
plt.ylabel('Computation time [s]',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join('Theory','results','computation_time.pdf'))

df.columns

df_melted = df.melt(id_vars=['Model','Patches','time'],var_name='Metric',value_name='Value')

model_long_list = []
for a,b in zip(df_melted['Model'].values,df_melted['Patches'].values):
    if not np.isnan(b):
        model_long_list.append(a+'_'+str(int(b)))
    else:
        model_long_list.append(a)

df_melted['Model_long'] = model_long_list


df_melted.loc[:10]
fig=plt.figure()
fig.set_size_inches(8,6)
sns.barplot(x='Model_long',y='Value',hue='Metric',data=df_melted[df_melted['Patches']!= 1])
plt.legend(fontsize=15,loc='upper left')
plt.xlabel('Model',fontsize=20)
plt.ylabel('Metric value',fontsize=20)
plt.xticks(fontsize=15,rotation=45)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join('Theory','results','metrics.pdf'))
