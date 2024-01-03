import torch, math, argparse, os, pickle
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from vae import *


def load_model(args):
    model = HierVAE(args).cuda()
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

    start = 0
    for i in range(args.epoch):
        if os.path.exists('%s/ckpt/%d.model' % (args.savedir, i)):
            start += 1

    if start > 0:
        model_state, optimizer_state, scheduler_state = torch.load('%s/ckpt/%d.model' % (args.savedir, start - 1))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
    return model, optimizer, scheduler, start


def main(args):
    torch.manual_seed(args.seed)

    vocab = [x.strip("\r\n ").split() for x in open('data/vocab_acid.txt')] 
    args.vocab_aci = PairVocab(vocab)
    vocab = [x.strip("\r\n ").split() for x in open('data/vocab_epoxide.txt')] 
    args.vocab_epo = PairVocab(vocab)

    model, optimizer, scheduler, start = load_model(args)

    total_iter = 31219

    with open('data/tensor/test.pkl', 'rb') as f:
        batches_test = pickle.load(f)
    
    meters = np.zeros(13)
    meters_cnt = 0
    train_losses = np.zeros(5)
    for epoch in range(start, args.epoch):
        beta = args.beta
        
        dataset = DataFolder(seed=epoch)

        log = open('%s/ckpt/%d.log' % (args.savedir, epoch), 'a')
        log.write('Epoch: %d, learning rate: %.6f\n' % (epoch + 1, scheduler.get_last_lr()[0]))
        log.flush()

        for it, batch in enumerate(dataset):
            n_data = len(batch[0])
            batch_aci = batch[1]
            batch_epo = batch[3]
            model.zero_grad()
            loss, kl_div, loss_aci, loss_epo, loss_prop, wacc_aci, iacc_aci, tacc_aci, sacc_aci, wacc_epo, iacc_epo, tacc_epo, sacc_epo = model((batch_aci, batch_epo), beta=beta, beta_prop=args.beta_prop)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            meters = meters + np.array([loss.item(), kl_div, loss_aci, loss_epo, loss_prop, wacc_aci * 100, iacc_aci * 100, tacc_aci * 100, sacc_aci * 100, wacc_epo * 100, iacc_epo * 100, tacc_epo * 100, sacc_epo * 100])
            meters_cnt += 1

            train_losses = train_losses + np.array([n_data, n_data * kl_div, loss_aci, loss_epo, n_data * loss_prop])

            if (it + 1) % args.print_iter == 0 or (it + 1) == total_iter:
                meters /= meters_cnt
                log.write("[%d] beta: %.3f, beta_prop: %.3f, loss: %.3f, KL: %.2f, loss_aci: %.2f, loss_epo: %.2f, loss_prop: %.2f, Word_aci: %.2f, %.2f, Topo_aci: %.2f, Assm_aci: %.2f, Word_epo: %.2f, %.2f, Topo_epo: %.2f, Assm_epo: %.2f\n" % 
                          (it+1, beta, args.beta_prop, meters[0], meters[1], meters[2], meters[3], meters[4], meters[5], meters[6], meters[7], meters[8], meters[9], meters[10], meters[11], meters[12]))
                log.flush()
                meters *= 0
                meters_cnt = 0
        scheduler.step()

        log.close()

        ckpt = (model.state_dict(), optimizer.state_dict(), scheduler.state_dict())
        torch.save(ckpt, '%s/ckpt/%d.model' % (args.savedir, epoch))

        with open('%s/train_loss.csv' % (args.savedir), 'a') as f:
            f.write('%d,%d,%.4f,%.4f,%.4f\n' % 
                    (epoch, train_losses[0], train_losses[1], train_losses[2], train_losses[3], train_losses[4]))
        train_losses *= 0
        
        model.eval()
        test_losses = np.zeros(5)
        with torch.no_grad():
            for batch in batches_test:
                n_data = len(batch[0])
                batch_aci = batch[1]
                batch_epo = batch[3]
                loss, kl_div, loss_aci, loss_epo, loss_prop, wacc_aci, iacc_aci, tacc_aci, sacc_aci, wacc_epo, iacc_epo, tacc_epo, sacc_epo = model((batch_aci, batch_epo), beta=beta, beta_prop=args.beta_prop)
                test_losses = test_losses + np.array([n_data, n_data * kl_div, loss_aci, loss_epo, n_data * loss_prop])
            with open('%s/test_loss.csv' % (args.savedir), 'a') as f:
                    f.write('%d,%d,%.4f,%.4f,%.4f\n' % 
                            (epoch, test_losses[0], test_losses[1], test_losses[2], test_losses[3], test_losses[4]))
            test_losses *= 0
        model.train()


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

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--clip_norm', type=float, default=20.0)
    parser.add_argument('--beta', type=float, default=0.005)
    parser.add_argument('--beta_prop', type=float, default=1)

    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--anneal_rate', type=float, default=1)
    parser.add_argument('--print_iter', type=int, default=50)

    args = parser.parse_args()

    main(args)
