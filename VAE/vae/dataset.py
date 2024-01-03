import pickle, random, gc


class DataFolder(object):

    def __init__(self, seed=0):
        self.data_files = ['data/tensor/train_'+str(i)+'.pkl' for i in range(31)]
        self.seed = seed

    def __len__(self):
        return 31219

    def __iter__(self):
        for fn in self.data_files:
            with open(fn, 'rb') as f:
                batches = pickle.load(f)

            random.Random(self.seed).shuffle(batches)
            for batch in batches:
                yield batch

            del batches
            gc.collect()


class DataFolder_prop(object):

    def __init__(self, seed=0):
        self.data_files = ['data/tensor/prop_train.pkl']
        self.seed = seed

    def __len__(self):
        return 232

    def __iter__(self):
        for fn in self.data_files:
            with open(fn, 'rb') as f:
                batches = pickle.load(f)

            random.Random(self.seed).shuffle(batches)
            for batch in batches:
                yield batch

            del batches
            gc.collect()
