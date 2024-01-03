from rdkit import Chem
import gpflow
from gpflow.mean_functions import Constant
from gpflow.utilities import positive
from gpflow.utilities.ops import broadcasting_elementwise
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem, RDConfig
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import sys, os
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


def vitrimerize(acid, epoxide):
    acid_mol = Chem.MolFromSmiles(acid)
    epoxide_mol = Chem.MolFromSmiles(epoxide)
    rxn1 = AllChem.ReactionFromSmarts('[CX3:1](=O)[OX2H1:2]>>[CX3:1](=O)[OX2:2][*]')
    acid_mol = rxn1.RunReactants((acid_mol, ))[0][0]
    Chem.SanitizeMol(acid_mol)
    rxn2 = AllChem.ReactionFromSmarts('[OD2r3:1]1[#6D2r3:2][#6r3:3]1>>[#6:3]([OD2:1])[#6D2:2][*]')
    epoxide_mol = rxn2.RunReactants((epoxide_mol, ))[0][0]
    Chem.SanitizeMol(epoxide_mol)
    rxn = AllChem.ReactionFromSmarts('[CX3:1](=O)[OX2H1:2].[OD2r3:3]1[#6D2r3:4][#6r3:5]1>>[CX3:1](=O)[OX2:2][#6D2:4][#6:5]([OD2:3])')
    vitrimer_mol = rxn.RunReactants((acid_mol, epoxide_mol))[0][0]
    Chem.SanitizeMol(vitrimer_mol)
    vitrimer = Chem.MolToSmiles(vitrimer_mol)
    return Chem.CanonSmiles(vitrimer)


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


def GP_calibration(acid, epoxide, tg_md_vitrimer):
    smiles, tg_exp, tg_md = load_data()
    mols = [AllChem.MolFromSmiles(s) for s in smiles]
    X_train = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048) for mol in mols]
    X_train = np.asarray(X_train)
    y_train = tg_exp - tg_md
    y_train = y_train.reshape(-1, 1)
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)

    smiles_list_test = [vitrimerize(acid[i], epoxide[i]) for i in range(len(acid))]
    mols = [AllChem.MolFromSmiles(smiles) for smiles in smiles_list_test]
    X_test = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048) for mol in mols]
    X_test = np.asarray(X_test)

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
    y_pred = y_scaler.inverse_transform(y_pred.numpy()).squeeze()
    return tg_md_vitrimer + y_pred


df = pd.read_csv('tg_vitrimer_MD.csv')
acid = df['acid'].to_list()
epoxide = df['epoxide'].to_list()
tg_md = df['tg'].to_list()
tg = GP_calibration(acid, epoxide, tg_md)
df = pd.DataFrame({'acid': acid, 'epoxide': epoxide, 'tg': tg})
df.to_csv('tg_vitrimer_calibrated.csv', index=False)
