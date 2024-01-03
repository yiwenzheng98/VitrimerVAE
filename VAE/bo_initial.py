import argparse, torch, pickle, os
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


def main(args, seed):
    torch.manual_seed(seed)

    vocab = [x.strip("\r\n ").split() for x in open('data/vocab_acid.txt')] 
    args.vocab_aci = PairVocab(vocab)
    vocab = [x.strip("\r\n ").split() for x in open('data/vocab_epoxide.txt')] 
    args.vocab_epo = PairVocab(vocab)

    model = HierVAE(args).cuda()
    model.load_state_dict(torch.load('%s/ckpt/prop49.model' % (args.savedir))[0])
    model.eval()

    with open('data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('%s/pca.pkl' % (args.savedir), 'rb') as f:
        pca = pickle.load(f)
    
    vitrimer = []
    for _ in range(30):
        aci_dec, epo_dec = model.sample(50)
        for i in range(50):
            if check_acid(aci_dec[i]) and check_epoxide(epo_dec[i]) and (aci_dec[i], epo_dec[i]) not in vitrimer:
                vitrimer.append((aci_dec[i], epo_dec[i]))
        if len(vitrimer) >= 1000:
            break

    acid, epoxide = zip(*vitrimer[:1000])
    acid = list(acid)
    epoxide = list(epoxide)

    acid_batches = [acid[i : i + 32] for i in range(0, 1000, 32)]
    epoxide_batches = [epoxide[i : i + 32] for i in range(0, 1000, 32)]
    z = []
    tg_norm = []
    acid = []
    epoxide = []
    for i in range(len(acid_batches)):
        tensors = tensorize(acid_batches[i], epoxide_batches[i], args.vocab_aci, args.vocab_epo)
        if tensors:
            z_recon, _ = model.encode(tensors)
            tg_recon = model.predict(z_recon).squeeze()
            z.append(z_recon.detach().cpu().numpy())
            tg_norm.extend(tg_recon.detach().cpu().tolist())
            acid.extend(acid_batches[i])
            epoxide.extend(epoxide_batches[i])
    z = np.vstack(z)
    z_pca = pca.transform(z)
    tg_norm = np.array(tg_norm)
    tg = scaler.inverse_transform(tg_norm.reshape(-1, 1)).squeeze()
    d = {'acid': acid, 'epoxide': epoxide, 'tg': tg, 'tg_norm': tg_norm, 'pca1': z_pca[:, 0], 'pca2': z_pca[:, 1]}
    for i in range(args.latent_size):
        d['z' + str(i)] = z[:, i]
    df = pd.DataFrame(d)
    df.to_csv('%s/bo_initial_%d.csv' % (args.savedir, seed), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, required=True)
    parser.add_argument('--atom_vocab', default=common_atom_vocab)

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

    for i in range(1, 11):
        if not os.path.isfile('%s/bo_initial_%d.csv' % (args.savedir, i)):
            main(args, i)

