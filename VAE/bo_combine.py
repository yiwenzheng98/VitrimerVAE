import pandas as pd
import numpy as np


target = 1
if target == 1:
    target_num = 248
elif target == 2:
    target_num = 373
else:
    target_num = 999
name = 'bo' + str(target)
vitrimer_all = []
tg_all = []
pca1_all = []
pca2_all = []
for i in range(1, 11):
    df = pd.read_csv('results/%s_%d.csv' % (name, i))
    acid = df['acid'].to_list()
    epoxide = df['epoxide'].to_list()
    tg = df['tg_pred'].to_list()
    pca1 = df['pca1'].to_list()
    pca2 = df['pca2'].to_list()
    for j in range(len(acid)):
        if (acid[j], epoxide[j]) not in vitrimer_all:
            vitrimer_all.append((acid[j], epoxide[j]))
            tg_all.append(tg[j])
            pca1_all.append(pca1[j])
            pca2_all.append(pca2[j])
acid_all, epoxide_all = zip(*vitrimer_all)
acid_all = list(acid_all)
epoxide_all = list(epoxide_all)
acid_all = np.array(acid_all)
epoxide_all = np.array(epoxide_all)
tg_all = np.array(tg_all)
pca1_all = np.array(pca1_all)
pca2_all = np.array(pca2_all)
ind = np.argsort(np.abs(tg_all - target_num))[:200]
df = pd.DataFrame({'acid': acid_all[ind], 'epoxide': epoxide_all[ind], 'tg_pred': tg_all[ind], 'pca1': pca1_all[ind], 'pca2': pca2_all[ind]})
df.to_csv('results/%s.csv' % name, index=False, float_format='%.4f')
