import argparse, pickle, torch
import pandas as pd
import numpy as np
from vae import *
from sklearn.decomposition import PCA


def extract_latent_vector(batches, model):
    acid_all = []
    epoxide_all = []
    tg_all = []
    tg_norm_all = []
    z_all = []
    for batch in batches:
        acid, tensor_aci_batch, epoxide, tensor_epo_batch, tg, tg_norm = batch
        z = model.encode((tensor_aci_batch, tensor_epo_batch))[0].cpu().detach().numpy()
        acid_all.extend(acid)
        epoxide_all.extend(epoxide)
        tg_all.extend(tg)
        tg_norm_all.extend(tg_norm)
        z_all.append(z)
    z_all = np.vstack(z_all)        
    return acid_all, epoxide_all, tg_all, tg_norm_all, z_all


def save_latent_vector(acid, epoxide, tg, tg_norm, z, pca, name):
    d = {'acid': acid, 'epoxide': epoxide, 'tg': tg, 'tg_norm': tg_norm, 'pca1': pca[:, 0], 'pca2': pca[:, 1]}
    for i in range(args.latent_size):
        d['z' + str(i)] = z[:, i]
    df = pd.DataFrame(d)
    df.to_csv('%s/latent_vector_%s.csv' % (args.savedir, name), index=False)


def main(args):
    torch.manual_seed(args.seed)

    vocab = [x.strip("\r\n ").split() for x in open('data/vocab_acid.txt')] 
    args.vocab_aci = PairVocab(vocab)
    vocab = [x.strip("\r\n ").split() for x in open('data/vocab_epoxide.txt')] 
    args.vocab_epo = PairVocab(vocab)

    # after joint training
    model = HierVAE(args).cuda()
    model.load_state_dict(torch.load('%s/ckpt/step2.model' % (args.savedir))[0])
    model.eval()

    with open('data/tensor/prop_train.pkl', 'rb') as f:
        batches_train = pickle.load(f)
    acid_train, epoxide_train, tg_train, tg_norm_train, z_train = extract_latent_vector(batches_train, model)
    
    with open('data/tensor/test.pkl', 'rb') as f:
        batches_test = pickle.load(f)
    acid_test, epoxide_test, tg_test, tg_norm_test, z_test = extract_latent_vector(batches_test, model)
    
    pca = PCA(n_components=2)
    pca.fit(z_train)
    pca_train = pca.transform(z_train)
    pca_test = pca.transform(z_test)

    with open('%s/pca.pkl' % (args.savedir), 'wb') as f:
        pickle.dump(pca, f, pickle.HIGHEST_PROTOCOL)

    save_latent_vector(acid_train, epoxide_train, tg_train, tg_norm_train, z_train, pca_train, 'train_joint')
    save_latent_vector(acid_test, epoxide_test, tg_test, tg_norm_test, z_test, pca_test, 'test_joint')
    
    # before joint training
    model = HierVAE(args).cuda()
    model.load_state_dict(torch.load('%s/ckpt/step1.model' % (args.savedir))[0])
    model.eval()

    with open('data/tensor/prop_train.pkl', 'rb') as f:
        batches_train = pickle.load(f)
    acid_train, epoxide_train, tg_train, tg_norm_train, z_train = extract_latent_vector(batches_train, model)
    
    with open('data/tensor/test.pkl', 'rb') as f:
        batches_test = pickle.load(f)
    acid_test, epoxide_test, tg_test, tg_norm_test, z_test = extract_latent_vector(batches_test, model)
    
    pca = PCA(n_components=2)
    pca.fit(z_train)
    pca_train = pca.transform(z_train)
    pca_test = pca.transform(z_test)

    save_latent_vector(acid_train, epoxide_train, tg_train, tg_norm_train, z_train, pca_train, 'train')
    save_latent_vector(acid_test, epoxide_test, tg_test, tg_norm_test, z_test, pca_test, 'test')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, required=True)
    parser.add_argument('--atom_vocab', default=common_atom_vocab)
    parser.add_argument('--seed', type=int, default=5)

    parser.add_argument('--rnn_type', type=str, default='LSTM')
    parser.add_argument('--hidden_size', type=int, default=250)
    parser.add_argument('--embed_size', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--latent_size', type=int, default=128)
    parser.add_argument('--acid_size', type=int, default=112)
    parser.add_argument('--epoxide_size', type=int, default=112)
    parser.add_argument('--share_size', type=int, default=96)
    parser.add_argument('--depthT', type=int, default=15)
    parser.add_argument('--depthG', type=int, default=15)
    parser.add_argument('--diterT', type=int, default=1)
    parser.add_argument('--diterG', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--prop_hidden_size', type=int, default=64)

    args = parser.parse_args()

    main(args)
