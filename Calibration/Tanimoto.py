import gpflow
from gpflow.mean_functions import Constant
from gpflow.utilities import positive
from gpflow.utilities.ops import broadcasting_elementwise
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tqdm import tqdm


def load_data():
    df = pd.read_csv('tg_calibration.csv')
    smiles = df['smiles'].to_numpy()
    tg_exp = df['tg_exp'].to_numpy()
    tg_md = df['tg_md'].to_numpy()
    ind = np.where(tg_md > 0)
    return smiles[ind], tg_exp[ind], tg_md[ind]


class Tanimoto(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__()
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        Xs = tf.reduce_sum(tf.square(X), axis=-1) 
        X2s = tf.reduce_sum(tf.square(X2), axis=-1) 
        outer_product = tf.tensordot(X, X2, [[-1], [-1]]) 

        denominator = -outer_product + broadcasting_elementwise(tf.add, Xs, X2s)

        return self.variance * outer_product/denominator

    def K_diag(self, X):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))


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

    m = None
    def objective_closure():
        return -m.log_marginal_likelihood()

    k = Tanimoto()
    m = gpflow.models.GPR(data=(X_train, y_train), mean_function=Constant(np.mean(y_train)), kernel=k, noise_variance=1)

    opt = gpflow.optimizers.Scipy()
    opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=100))

    y_pred, _ = m.predict_f(X_test)
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
    df.to_csv('LOOCV_Tanimoto.csv', index=False)


LOOCV()
