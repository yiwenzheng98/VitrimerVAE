import random, pickle, argparse, torch
import pandas as pd
import numpy as np
from rdkit import Chem
from functools import partial
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from vae import MolGraph, common_atom_vocab, PairVocab


def canon_smiles(smiles):
    try:
        c_smiles = Chem.CanonSmiles(smiles)
    except:
        c_smiles = ''
    return c_smiles


def canon_smiles_list(smiles_list):
    return [canon_smiles(smiles) for smiles in smiles_list]


def to_numpy(tensors):
    convert = lambda x : x.numpy() if type(x) is torch.Tensor else x
    a,b,c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c


def tensorize(mol_batch, vocab):
    try:
        x = MolGraph.tensorize(mol_batch, vocab, common_atom_vocab)
        return [mol_batch, to_numpy(x)]
    except:
        print('Unsuccessful tensorize: the whole batch returned to None')
        print(mol_batch)
        return [mol_batch, None]


def normalize(x, scaler):
    x_copy = np.array(x).reshape(-1, 1)
    x_norm = scaler.transform(x_copy)
    x_norm = list(x_norm.squeeze())
    return x_norm


def write(data, name, split=False):
    if split:
        num_splits = len(data) // 1000
        le = (len(data) + num_splits - 1) // num_splits
        for split_id in range(num_splits):
            st = split_id * le
            sub_data = data[st : st + le]
            with open('data/tensor/%s_%d.pkl' % (name, split_id), 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open('data/tensor/%s.pkl' % (name), 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def process(dataset, args):
    pool = Pool(args.ncpu)

    with open('data/vocab_acid.txt') as f:
        vocab = [x.strip("\r\n ").split() for x in f]
    vocab_acid = PairVocab(vocab, cuda=False)

    with open('data/vocab_epoxide.txt') as f:
        vocab = [x.strip("\r\n ").split() for x in f]
    vocab_epoxide = PairVocab(vocab, cuda=False)

    # load data
    df = pd.read_csv('data/%s.csv' % dataset)
    acid = canon_smiles_list(df['acid'].to_list())
    epoxide = canon_smiles_list(df['epoxide'].to_list())
    vitrimer = list(zip(acid, epoxide))
    random.shuffle(vitrimer)
    acid, epoxide = zip(*vitrimer)
    acid = list(acid)
    epoxide = list(epoxide)

    # process acid
    batches = [acid[i : i + args.batch_size] for i in range(0, len(acid), args.batch_size)]
    func = partial(tensorize, vocab = vocab_acid)
    data_acid = pool.map(func, batches)

    # process epoxide
    batches = [epoxide[i : i + args.batch_size] for i in range(0, len(epoxide), args.batch_size)]
    func = partial(tensorize, vocab = vocab_epoxide)
    data_epoxide = pool.map(func, batches)

    # combine data
    data = [data_acid[i] + data_epoxide[i] for i in range(len(data_acid))]
    return data


def process_prop(dataset, args, scaler):
    pool = Pool(args.ncpu)

    with open('data/vocab_acid.txt') as f:
        vocab = [x.strip("\r\n ").split() for x in f]
    vocab_acid = PairVocab(vocab, cuda=False)

    with open('data/vocab_epoxide.txt') as f:
        vocab = [x.strip("\r\n ").split() for x in f]
    vocab_epoxide = PairVocab(vocab, cuda=False)

    # load data
    df = pd.read_csv('data/%s.csv' % dataset)
    acid = canon_smiles_list(df['acid'].to_list())
    epoxide = canon_smiles_list(df['epoxide'].to_list())
    tg = df['tg'].to_list()
    vitrimer = list(zip(acid, epoxide, tg))
    random.shuffle(vitrimer)
    acid, epoxide, tg = zip(*vitrimer)
    acid = list(acid)
    epoxide = list(epoxide)
    tg = list(tg)
    tg_norm = normalize(tg, scaler)

    # process acid
    batches = [acid[i : i + args.batch_size] for i in range(0, len(acid), args.batch_size)]
    func = partial(tensorize, vocab = vocab_acid)
    data_acid = pool.map(func, batches)

    # process epoxide
    batches = [epoxide[i : i + args.batch_size] for i in range(0, len(epoxide), args.batch_size)]
    func = partial(tensorize, vocab = vocab_epoxide)
    data_epoxide = pool.map(func, batches)

    # batch tg as well
    data_tg = [[tg[i : i + args.batch_size]] for i in range(0, len(tg), args.batch_size)]
    data_tg_norm = [[tg_norm[i : i + args.batch_size]] for i in range(0, len(tg_norm), args.batch_size)]

    # combine data
    data = [data_acid[i] + data_epoxide[i] + data_tg[i] + data_tg_norm[i] for i in range(len(data_acid))]
    return data

        
def main(args):
    random.seed(args.seed)

    # normalize prop training set and get the scaler
    df = pd.read_csv('data/prop_train.csv')
    tg = df['tg'].to_numpy().reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(tg)
    with open('data/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f, pickle.HIGHEST_PROTOCOL)

    data_train = process('train', args)
    data_prop_train = process_prop('prop_train', args, scaler)
    data_prop_test = process_prop('test', args, scaler)

    data = data_train + data_prop_train
    random.shuffle(data)
    write(data, 'train', split=True)
    write(data_prop_train, 'prop_train', split=False)
    write(data_prop_test, 'test', split=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--ncpu', type=int, default=4)
    parser.add_argument('--seed', type=int, default=5)
    args = parser.parse_args()

    main(args)
