import argparse, torch, pickle
import pandas as pd
from vae import *


def main(args):
    torch.manual_seed(args.seed)

    vocab = [x.strip("\r\n ").split() for x in open('data/vocab_acid.txt')] 
    args.vocab_aci = PairVocab(vocab)
    vocab = [x.strip("\r\n ").split() for x in open('data/vocab_epoxide.txt')] 
    args.vocab_epo = PairVocab(vocab)

    model = HierVAE(args).cuda()
    model.load_state_dict(torch.load('%s/ckpt/step2.model' % (args.savedir))[0])
    model.eval()

    with open('data/tensor/test.pkl', 'rb') as f:
        batches = pickle.load(f)
    b = batches[0]
    aci = b[0]
    epo = b[2]
    tensor_aci = b[1]
    tensor_epo = b[3]
    aci_dec, epo_dec = model.reconstruct((tensor_aci, tensor_epo))
    aci_dec = canon_smiles_list(aci_dec)
    epo_dec = canon_smiles_list(epo_dec)

    pd.DataFrame({'aci': aci, 'epo': epo, 'aci_dec': aci_dec, 'epo_dec': epo_dec}).to_csv('%s/reconstruct.csv' % (args.savedir), index=False)
    


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
