import numpy as np
import pandas as pd


df = pd.read_csv('tg_calibration.csv')
tg_exp = df['tg_exp'].to_numpy()
tg_md = df['tg_md'].to_numpy()
n = len(tg_exp)
tg_pred = []
for i in range(n):
    x = np.delete(tg_md, i)
    y = np.delete(tg_exp, i)
    p = np.polyfit(x, y, 1)
    tg_pred.append(np.polyval(p, tg_md[i]))
df = pd.DataFrame({'tg_exp': tg_exp, 'tg_cal': tg_pred})
df.to_csv('LOOCV_linear.csv', index=False, float_format='%.4f')
