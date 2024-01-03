import argparse 
import pandas as pd
from multiprocessing import Pool
from vae import *


def process(data):
    vocab = set()
    for line in data:
        s = line.strip("\r\n ")
        hmol = MolGraph(s)
        for node,attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            vocab.add( attr['label'] )
            for i,s in attr['inter_label']:
                vocab.add( (smiles, s) )
    return vocab


def main(args):
    df = pd.read_csv('data/train.csv')
    acid = df['acid'].to_list()
    epoxide = df['epoxide'].to_list()
    acid = list(set(acid))
    epoxide = list(set(epoxide))

    batch_size = len(acid) // args.ncpu + 1
    batches = [acid[i : i + batch_size] for i in range(0, len(acid), batch_size)]

    pool = Pool(args.ncpu)
    vocab_list = pool.map(process, batches)
    vocab = [(x,y) for vocab in vocab_list for x,y in vocab]
    vocab = list(set(vocab))

    with open('data/vocab_acid.txt', 'w') as f:
        for x,y in sorted(vocab):
            f.write(x + ' ' + y + '\n')

    batch_size = len(epoxide) // args.ncpu + 1
    batches = [epoxide[i : i + batch_size] for i in range(0, len(epoxide), batch_size)]

    pool = Pool(args.ncpu)
    vocab_list = pool.map(process, batches)
    vocab = [(x,y) for vocab in vocab_list for x,y in vocab]
    vocab = list(set(vocab))

    with open('data/vocab_epoxide.txt', 'w') as f:
        for x,y in sorted(vocab):
            f.write(x + ' ' + y + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncpu', type=int, default=1)
    args = parser.parse_args()
    
    main(args)
    