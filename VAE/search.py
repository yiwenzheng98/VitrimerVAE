import argparse, torch, random, pickle
import numpy as np
import pandas as pd
from vae import *


def to_numpy(tensors):
    convert = lambda x : x.numpy() if type(x) is torch.Tensor else x
    a,b,c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c


def tensorize(acid, epoxide, vocab_aci, vocab_epo):
    try:
        x_aci = MolGraph.tensorize(acid, vocab_aci, common_atom_vocab)
        x_epo = MolGraph.tensorize(epoxide, vocab_epo, common_atom_vocab)
        return to_numpy(x_aci), to_numpy(x_epo)
    except:
        return None


def decode(z, z_origin, acid_origin, epoxide_origin, z_pca_origin, tg_origin, model, pca, scaler, axis):
    z_tensor = torch.tensor(z).float().cuda()
    acid, epoxide = model.decode(z_tensor)
    tg_norm = model.predict(z_tensor).cpu().detach().numpy().reshape(-1, 1)
    tg = scaler.inverse_transform(tg_norm).squeeze()
    vitrimer = list(zip(acid, epoxide))
    acid_all = [acid_origin]
    epoxide_all = [epoxide_origin]
    z_all = []
    tg_all = [tg_origin]
    dist = [0]
    for i, v in enumerate(vitrimer):
        if check_acid(v[0]) and check_epoxide(v[1]):
            if axis == '1' and v[0] not in acid_all:
                acid_all.append(v[0])
                epoxide_all.append(v[1])
                z_all.append(z[i, :])
                tg_all.append(tg[i])
                dist.append(np.linalg.norm(z_origin - z[i, :]))
            elif axis == '3' and v[0] not in acid_all and v[1] not in epoxide_all:
                acid_all.append(v[0])
                epoxide_all.append(v[1])
                z_all.append(z[i, :])
                tg_all.append(tg[i])
                dist.append(np.linalg.norm(z_origin - z[i, :]))
            elif axis == '2' and v[1] not in epoxide_all:
                acid_all.append(v[0])
                epoxide_all.append(v[1])
                z_all.append(z[i, :])
                tg_all.append(tg[i])
                dist.append(np.linalg.norm(z_origin - z[i, :]))
    z_pca_all = np.vstack([z_pca_origin, pca.transform(np.vstack(z_all))])
    return acid_all, epoxide_all, z_pca_all, dist, tg_all


def main(acid, epoxide, args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

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

    # encode vitrimer to latent vector z
    tensor_aci, tensor_epo = tensorize([acid], [epoxide], args.vocab_aci, args.vocab_epo)
    z, _ = model.encode((tensor_aci, tensor_epo))
    tg = scaler.inverse_transform(model.predict(z).cpu().detach().numpy().reshape(-1, 1)).item()
    z = z.cpu().detach().numpy()
    z1 = z[:, :(args.latent_size - args.epoxide_size)]
    z2 = z[:, (args.latent_size - args.epoxide_size):(args.latent_size - args.epoxide_size + args.share_size)]
    z3 = z[:, (args.latent_size - args.epoxide_size + args.share_size):]
    z_pca = pca.transform(z)
    
    # add noise in 3 axes: acid-only (z[:8]), epoxide-only (z[56:]), and both (z)
    max_noise = 20
    n = 100

    z_neighbor1 = []
    for _ in range(n):
        noise_level = random.uniform(0, 1) * max_noise
        noise_dir = np.random.normal(0, 1, size=z1.shape)
        noise_dir /= np.linalg.norm(noise_dir)
        noise = noise_dir * noise_level
        z_neighbor1.append(np.concatenate((z1 + noise, z2, z3), axis=1))
    z_neighbor1 = np.vstack(z_neighbor1)

    z_neighbor2 = []
    for _ in range(n):
        noise_level = random.uniform(0, 1) * max_noise
        noise_dir = np.random.normal(0, 1, size=z3.shape)
        noise_dir /= np.linalg.norm(noise_dir)
        noise = noise_dir * noise_level
        z_neighbor2.append(np.concatenate((z1, z2, z3 + noise), axis=1))
    z_neighbor2 = np.vstack(z_neighbor2)

    z_neighbor3 = []
    for _ in range(n):
        noise_level = random.uniform(0, 1) * max_noise
        noise_dir = np.random.normal(0, 1, size=z.shape)
        noise_dir /= np.linalg.norm(noise_dir)
        noise = noise_dir * noise_level
        z_neighbor3.append(z + noise)
    z_neighbor3 = np.vstack(z_neighbor3)

    acid1, epoxide1, pca1, dist1, tg1 = decode(z_neighbor1, z, acid, epoxide, z_pca, tg, model, pca, scaler, '1')
    df1 = pd.DataFrame({'acid': acid1, 'epoxide': epoxide1, 'pca1': pca1[:, 0], 'pca2': pca1[:, 1], 'dist': dist1, 'tg': tg1})
    df1 = df1.sort_values(by='dist')
    df1.to_csv('%s/search1.csv' % (args.savedir), index=False, float_format='%.4f')

    acid2, epoxide2, pca2, dist2, tg2 = decode(z_neighbor2, z, acid, epoxide, z_pca, tg, model, pca, scaler, '2')
    df2 = pd.DataFrame({'acid': acid2, 'epoxide': epoxide2, 'pca1': pca2[:, 0], 'pca2': pca2[:, 1], 'dist': dist2, 'tg': tg2})
    df2 = df2.sort_values(by='dist')
    df2.to_csv('%s/search2.csv' % (args.savedir), index=False, float_format='%.4f')

    acid3, epoxide3, pca3, dist3, tg3 = decode(z_neighbor3, z, acid, epoxide, z_pca, tg, model, pca, scaler, '3')
    df3 = pd.DataFrame({'acid': acid3, 'epoxide': epoxide3, 'pca1': pca3[:, 0], 'pca2': pca3[:, 1], 'dist': dist3, 'tg': tg3})
    df3 = df3.sort_values(by='dist')
    df3.to_csv('%s/search3.csv' % (args.savedir), index=False, float_format='%.4f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--savedir', type=str, required=True)
    parser.add_argument('--atom_vocab', default=common_atom_vocab)
    parser.add_argument('--seed', type=int, default=1)

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

    # adipic acid + DGEBA
    acid = 'O=C(O)CCCCC(=O)O'
    epoxide = 'CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1'
    main(acid, epoxide, args)
