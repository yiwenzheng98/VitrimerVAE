import argparse, torch, pickle
import numpy as np
import pandas as pd
from vae import *


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def decode(z, z_origin, model, pca, scaler):
    z_tensor = torch.tensor(z).float().cuda()
    acid, epoxide = model.decode(z_tensor)
    tg_norm = model.predict(z_tensor).cpu().detach().numpy().reshape(-1, 1)
    tg = scaler.inverse_transform(tg_norm).squeeze()
    vitrimer = list(zip(acid, epoxide))
    vitrimer_all = []
    tg_all = []
    z_all = []
    dist = []
    for i, v in enumerate(vitrimer):
        if check_acid(v[0]) and check_epoxide(v[1]) and v not in vitrimer_all:
            vitrimer_all.append(v)
            tg_all.append(tg[i])
            dist.append(np.linalg.norm(z_origin - z[i, :]))
            z_all.append(z[i, :])
    z_pca_all = pca.transform(np.vstack(z_all))
    acid_all, epoxide_all = zip(*vitrimer_all)
    acid_all=  list(acid_all)
    epoxide_all = list(epoxide_all)
    return acid_all, epoxide_all, z_pca_all, dist, tg_all


def spherical_interpolation(start, end, model, n_inter, pca, scaler):
    acid_start, epoxide_start, z_start, z_pca_start, tg_start = start
    acid_end, epoxide_end, z_end, z_pca_end, tg_end = end

    theta = angle_between(z_end, z_start)

    z_inter = []
    for i in range(n_inter):
        alpha = (i + 1) / (n_inter + 1)
        z = (z_end * np.sin(alpha * theta) + z_start * np.sin((1 - alpha) * theta)) / np.sin(theta)
        z_inter.append(z)
    z_inter = np.vstack(z_inter)

    acid, epoxide, z_pca, dist, tg = decode(z_inter, z_start, model, pca, scaler)
    acid.append(acid_start)
    acid.append(acid_end)
    epoxide.append(epoxide_start)
    epoxide.append(epoxide_end)
    tg.append(tg_start)
    tg.append(tg_end)
    z_pca = np.vstack([z_pca, z_pca_start, z_pca_end])
    dist.append(0)
    dist.append(np.linalg.norm(z_start - z_end))
    return acid, epoxide, z_pca, dist, tg


def linear_interpolation(start, end, model, n_inter, pca, scaler):
    acid_start, epoxide_start, z_start, z_pca_start, tg_start = start
    acid_end, epoxide_end, z_end, z_pca_end, tg_end = end

    z_inter = []
    for i in range(n_inter):
        alpha = (i + 1) / (n_inter + 1)
        z = z_end * alpha + z_start * (1 - alpha)
        z_inter.append(z)
    z_inter = np.vstack(z_inter)

    acid, epoxide, z_pca, dist, tg = decode(z_inter, z_start, model, pca, scaler)
    acid.append(acid_start)
    acid.append(acid_end)
    epoxide.append(epoxide_start)
    epoxide.append(epoxide_end)
    tg.append(tg_start)
    tg.append(tg_end)
    z_pca = np.vstack([z_pca, z_pca_start, z_pca_end])
    dist.append(0)
    dist.append(np.linalg.norm(z_start - z_end))
    return acid, epoxide, z_pca, dist, tg


def main(args):
    torch.manual_seed(args.seed)

    with open('%s/pca.pkl' % (args.savedir), 'rb') as f:
        pca = pickle.load(f)
    
    with open('data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    vocab = [x.strip("\r\n ").split() for x in open('data/vocab_acid.txt')] 
    args.vocab_aci = PairVocab(vocab)
    vocab = [x.strip("\r\n ").split() for x in open('data/vocab_epoxide.txt')] 
    args.vocab_epo = PairVocab(vocab)

    model = HierVAE(args).cuda()
    model.load_state_dict(torch.load('%s/ckpt/step2.model' % (args.savedir))[0])
    model.eval()

    df = pd.read_csv('%s/latent_vector_train_joint.csv' % (args.savedir))
    df = df.sort_values(by='tg')
    acid_all = df['acid'].to_numpy()
    epoxide_all = df['epoxide'].to_numpy()
    tg_all = df['tg'].to_numpy()
    z_all = df.loc[:, 'z0':'z'+str(args.latent_size-1)].to_numpy()
    pca1_all = df['pca1'].to_numpy()
    pca2_all = df['pca2'].to_numpy()
    
    start = (acid_all[0], epoxide_all[0], z_all[0, :], np.array([pca1_all[0], pca2_all[0]]), tg_all[0])
    end = (acid_all[-1], epoxide_all[-1], z_all[-1, :], np.array([pca1_all[-1], pca2_all[-1]]), tg_all[-1])
    if args.method == 'linear':
        acid, epoxide, z_pca, dist, tg = linear_interpolation(start, end, model, 20, pca, scaler)
    elif args.method == 'spherical':
        acid, epoxide, z_pca, dist, tg = spherical_interpolation(start, end, model, 20, pca, scaler)

    df = pd.DataFrame({'acid': acid, 'epoxide': epoxide, 'pca1': z_pca[:, 0], 'pca2': z_pca[:, 1], 'dist': dist, 'tg': tg})
    df = df.sort_values(by='dist')
    df.to_csv('%s/interpolation_%s.csv' % (args.savedir, args.method), index=False, float_format='%.4f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--savedir', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
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
