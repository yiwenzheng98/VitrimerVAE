import argparse, torch, pickle
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from vae import *


def reconstruction(model):
    with open('data/tensor/test.pkl', 'rb') as f:
        batches = pickle.load(f)
    acc = 0
    tot = 0
    for b in batches:
        aci = b[0]
        epo = b[2]
        tensor_aci = b[1]
        tensor_epo = b[3]
        aci_dec, epo_dec = model.reconstruct((tensor_aci, tensor_epo))
        aci_dec = canon_smiles_list(aci_dec)
        epo_dec = canon_smiles_list(epo_dec)
        for i in range(len(aci)):
            if aci[i] == aci_dec[i] and epo[i] == epo_dec[i]:
                acc += 1
            tot += 1
    return acc/tot


def sample(model, vitrimer_train):
    n_sample = 1000
    batch_size = 50
    valid = 0
    novel = 0
    unique = []
    tot = 0
    for _ in range(n_sample // batch_size):
        aci_dec, epo_dec = model.sample(batch_size)
        aci_dec = canon_smiles_list(aci_dec)
        epo_dec = canon_smiles_list(epo_dec)
        for i in range(batch_size):
            # sample validity
            # chemical validity (valency)
            # acid/epoxide validity (exactly two functional groups)
            if check_acid(aci_dec[i]) and check_epoxide(epo_dec[i]):
                valid += 1
            # sample novelty
            if (aci_dec[i], epo_dec[i]) not in vitrimer_train:
                novel += 1
            tot += 1
            unique.append((aci_dec[i], epo_dec[i]))
    return valid/tot, novel/tot, len(set(unique))/n_sample


def prediction(model):
    with open('data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    data = ['prop_train', 'test']
    r2 = []
    mae = []
    rmse = []
    for d in data:
        with open('data/tensor/%s.pkl' % d, 'rb') as f:
            batches = pickle.load(f)
        tg = []
        tg_pred = []
        for batch in batches:
            tensor_aci_batch = batch[1]
            tensor_epo_batch = batch[3]
            z = model.encode((tensor_aci_batch, tensor_epo_batch))[0]
            tg.extend(batch[4])
            tg_pred.extend(list(scaler.inverse_transform(model.predict(z).cpu().detach().numpy()).squeeze()))
        r2.append(r2_score(tg, tg_pred))
        mae.append(mean_absolute_error(tg, tg_pred))
        rmse.append(mean_squared_error(tg, tg_pred, squared=False))
    return r2[0], r2[1], mae[0], mae[1], rmse[0], rmse[1], tg, tg_pred


def main(args):
    torch.manual_seed(args.seed)

    vocab = [x.strip("\r\n ").split() for x in open('data/vocab_acid.txt')] 
    args.vocab_aci = PairVocab(vocab)
    vocab = [x.strip("\r\n ").split() for x in open('data/vocab_epoxide.txt')] 
    args.vocab_epo = PairVocab(vocab)

    df = pd.read_csv('data/train.csv')
    acid_train = df['acid'].to_list()
    epoxide_train = df['epoxide'].to_list()
    vitrimer_train = set(zip(acid_train, epoxide_train))

    n_epoch = 10
    with open('%s/metrics.csv' % (args.savedir), 'w') as f:
        f.write('epoch,recon,valid,novel,unique,r2_train,r2_test,mae_train,mae_test,rmse_train,rmse_test\n')
        f.flush()
        for i in range(n_epoch):
            model = HierVAE(args).cuda()
            model.load_state_dict(torch.load('%s/ckpt/%d.model' % (args.savedir, i))[0])
            model.eval()

            r2_train, r2_test, mae_train, mae_test, rmse_train, rmse_test, tg, tg_pred = prediction(model)
            recon = reconstruction(model)
            valid, novel, unique = sample(model, vitrimer_train)

            f.write('%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n' 
                    % (i+1, recon, valid, novel, unique, r2_train, r2_test, mae_train, mae_test, rmse_train, rmse_test))
            f.flush()

    n_epoch = 50
    with open('%s/metrics_joint.csv' % (args.savedir), 'w') as f:
        f.write('epoch,recon,valid,novel,unique,r2_train,r2_test,mae_train,mae_test,rmse_train,rmse_test\n')
        f.flush()
        for i in range(n_epoch):
            model = HierVAE(args).cuda()
            model.load_state_dict(torch.load('%s/ckpt/prop%d.model' % (args.savedir, i))[0])
            model.eval()

            r2_train, r2_test, mae_train, mae_test, rmse_train, rmse_test, tg, tg_pred = prediction(model)
            recon = reconstruction(model)
            valid, novel, unique = sample(model, vitrimer_train)

            f.write('%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n' 
                    % (i+1, recon, valid, novel, unique, r2_train, r2_test, mae_train, mae_test, rmse_train, rmse_test))
            f.flush()
    
    df = pd.DataFrame({'tg': tg, 'tg_pred': tg_pred})
    df.to_csv('%s/prediction.csv' % (args.savedir), index=False, float_format='%.4f')


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
