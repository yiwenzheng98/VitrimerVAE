import torch
import torch.nn as nn
from vae.encoder import HierMPNEncoder
from vae.decoder import HierMPNDecoder
from vae.nnutils import *


def make_cuda(tensors):
    tree_tensors, graph_tensors = tensors
    make_tensor = lambda x: x if type(x) is torch.Tensor else torch.tensor(x)
    tree_tensors = [make_tensor(x).cuda().long() for x in tree_tensors[:-1]] + [tree_tensors[-1]]
    graph_tensors = [make_tensor(x).cuda().long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
    return tree_tensors, graph_tensors


class HierVAE(nn.Module):
    def __init__(self, args):
        super(HierVAE, self).__init__()
        self.encoder_aci = HierMPNEncoder(args.vocab_aci, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size, args.depthT, args.depthG, args.dropout)
        self.decoder_aci = HierMPNDecoder(args.vocab_aci, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size, args.acid_size, args.diterT, args.diterG, args.dropout)
        self.encoder_aci.tie_embedding(self.decoder_aci.hmpn)
        self.encoder_epo = HierMPNEncoder(args.vocab_epo, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size, args.depthT, args.depthG, args.dropout)
        self.decoder_epo = HierMPNDecoder(args.vocab_epo, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size, args.epoxide_size, args.diterT, args.diterG, args.dropout)
        self.encoder_epo.tie_embedding(self.decoder_epo.hmpn)

        self.latent_size = args.latent_size # 64
        self.acid_size = args.acid_size # 56
        self.epoxide_size = args.epoxide_size # 56
        self.share_size = args.share_size # 48
        self.batch_size = args.batch_size

        self.R_mean_aci = nn.Linear(args.hidden_size, args.acid_size)
        self.R_var_aci = nn.Linear(args.hidden_size, args.acid_size)
        self.R_mean_epo = nn.Linear(args.hidden_size, args.epoxide_size)
        self.R_var_epo = nn.Linear(args.hidden_size, args.epoxide_size)

        self.propNN = nn.Sequential(
                nn.Linear(args.latent_size, args.prop_hidden_size),
                nn.Linear(args.prop_hidden_size, 1)
        )
        self.prop_loss = nn.MSELoss()

    # molecule to z
    def encode(self, batch, train=False):
        batch_aci = batch[0]
        batch_epo = batch[1]
        graphs_aci, tensors_aci, orders_aci = batch_aci
        graphs_epo, tensors_epo, orders_epo = batch_epo

        tree_tensors_aci, graph_tensors_aci = tensors_aci = make_cuda(tensors_aci)
        tree_tensors_epo, graph_tensors_epo = tensors_epo = make_cuda(tensors_epo)

        root_vecs_aci, _, _, _ = self.encoder_aci(tree_tensors_aci, graph_tensors_aci)
        root_vecs_epo, _, _, _ = self.encoder_epo(tree_tensors_epo, graph_tensors_epo)

        z_mean_aci = self.R_mean_aci(root_vecs_aci) # 56
        z_mean_aci_spec = z_mean_aci[:, :(self.latent_size - self.acid_size)] # 0:8, 8
        z_mean_aci_share = z_mean_aci[:, (self.latent_size - self.acid_size):] # 8:56, 48
        z_mean_epo = self.R_mean_epo(root_vecs_epo) # 56
        z_mean_epo_spec = z_mean_epo[:, :(self.latent_size - self.epoxide_size)] # 0:8, 8
        z_mean_epo_share = z_mean_epo[:, (self.latent_size - self.epoxide_size):] # 8:56, 48
        z_log_var_aci = -torch.abs(self.R_var_aci(root_vecs_aci)) # 56
        z_log_var_aci_spec = z_log_var_aci[:, :(self.latent_size - self.acid_size)] # 0:8, 8
        z_log_var_aci_share = z_log_var_aci[:, (self.latent_size - self.acid_size):] # 8:56, 48
        z_log_var_epo = -torch.abs(self.R_var_epo(root_vecs_epo)) # 56
        z_log_var_epo_spec = z_log_var_epo[:, :(self.latent_size - self.epoxide_size)] # 0:8, 8
        z_log_var_epo_share = z_log_var_epo[:, (self.latent_size - self.epoxide_size):] # 8:56, 48
        z_mean = torch.cat((z_mean_aci_spec, (z_mean_aci_share + z_mean_epo_share)/2, z_mean_epo_spec), dim=1) # 8 cat 48 cat 8, 64
        z_log_var = torch.cat((z_log_var_aci_spec, (z_log_var_aci_share + z_log_var_epo_share)/2, z_log_var_epo_spec), dim=1) # 8 cat 48 cat 8, 64
        if train:
            return z_mean, z_log_var, graphs_aci, tensors_aci, orders_aci, graphs_epo, tensors_epo, orders_epo
        return z_mean, z_log_var

    def repar_trick(self, z_mean, z_log_var):
        epsilon = torch.randn_like(z_mean).cuda()
        z = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z
    
    # z to molecule
    def decode(self, z):
        z_aci = z[:, :self.acid_size] # 0:56, 56
        z_epo = z[:, (self.latent_size - self.epoxide_size):] # 8:64, 56
        aci_dec = self.decoder_aci.decode((z_aci, z_aci, z_aci), greedy=True, max_decode_step=150)
        epo_dec = self.decoder_epo.decode((z_epo, z_epo, z_epo), greedy=True, max_decode_step=150)
        return aci_dec, epo_dec

    def sample(self, batch_size):
        z = torch.randn(batch_size, self.latent_size).cuda()
        return self.decode(z)

    def reconstruct(self, batch):
        z_mean, _ = self.encode(batch)
        aci_dec, epo_dec = self.decode(z_mean)
        return aci_dec, epo_dec
    
    def predict(self, z):
        return self.propNN(z)
    
    def forward(self, batch, beta, beta_prop, prop=False):
        z_mean, z_log_var, graphs_aci, tensors_aci, orders_aci, graphs_epo, tensors_epo, orders_epo = self.encode(batch, train=True)
        kl_div = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / len(orders_epo)
        z = self.repar_trick(z_mean, z_log_var)
        z_aci = z[:, :self.acid_size] # 0:56, 56
        z_epo = z[:, (self.latent_size - self.epoxide_size):] # 8:64, 56
        loss_aci, wacc_aci, iacc_aci, tacc_aci, sacc_aci = self.decoder_aci((z_aci, z_aci, z_aci), graphs_aci, tensors_aci, orders_aci)
        loss_epo, wacc_epo, iacc_epo, tacc_epo, sacc_epo = self.decoder_epo((z_epo, z_epo, z_epo), graphs_epo, tensors_epo, orders_epo)
        if prop:
            batch_prop = batch[2]
            batch_prop = torch.tensor(batch_prop).float().cuda()
            batch_prop_pred = self.propNN(z).squeeze()
            loss_prop = self.prop_loss(batch_prop_pred, batch_prop)
        else:
            loss_prop = torch.tensor(0).float().cuda()
        loss = loss_aci + loss_epo + beta * kl_div + beta_prop * loss_prop
        return loss, kl_div.item(), loss_aci.item(), loss_epo.item(), loss_prop.item(), wacc_aci, iacc_aci, tacc_aci, sacc_aci, wacc_epo, iacc_epo, tacc_epo, sacc_epo

