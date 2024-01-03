import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF
import tensorflow as tf
from tqdm import tqdm


def load_data():
    df = pd.read_csv('tg_calibration.csv')
    smiles = df['smiles'].to_numpy()
    tg_exp = df['tg_exp'].to_numpy()
    tg_md = df['tg_md'].to_numpy()
    ind = np.where(tg_md > 0)
    return smiles[ind], tg_exp[ind], tg_md[ind]


def transform_data(y_train, y_test):
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    return y_train_scaled, y_test_scaled, y_scaler


def GP_calibration(smiles, tg_exp, tg_md, test_idx):
    y = tg_exp - tg_md
    rdkit_mols = [AllChem.MolFromSmiles(s) for s in smiles]
    X = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048) for mol in rdkit_mols]
    X = np.asarray(X)

    X_train = np.delete(X, (test_idx), axis=0)
    y_train = np.delete(y, (test_idx), axis=0)
    X_test = X[test_idx, :]
    y_test = y[test_idx]

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1).T

    y_train, y_test, y_scaler = transform_data(y_train, y_test)

    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)

    kernel = RBF(length_scale=10)
    gp = gaussian_process.GaussianProcessRegressor(kernel = kernel)
    gp.fit(X_train, y_train)
    y_pred = gp.predict(X_test).reshape(-1, 1)
    
    y_pred = y_scaler.inverse_transform(y_pred)

    tg_exp_test = tg_exp[test_idx]
    tg_pred_test = tg_md[test_idx] + tf.squeeze(y_pred)
    tg_pred_test = tg_pred_test.numpy()

    return tg_exp_test, tg_pred_test


def LOOCV():
    smiles, tg_exp, tg_md = load_data()
    tg_gt = []
    tg_pred = []
    for i in tqdm(range(tg_exp.size)):
        tg_exp_test, tg_pred_test = GP_calibration(smiles, tg_exp, tg_md, i)
        tg_gt.append(tg_exp_test)
        tg_pred.append(tg_pred_test)
    tg_gt = np.array(tg_gt)
    tg_pred = np.array(tg_pred)
    
    df = pd.DataFrame({'tg_exp': tg_gt, 'tg_cal': tg_pred})
    df.to_csv('LOOCV_RBF.csv', index=False)


LOOCV()
