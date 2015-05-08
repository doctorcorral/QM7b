try:
    import cPickle as pickle
except:
    import pickle
import sys
import numpy as np

class qmDL(object):
    def __init__(self):
        pass

    def load(self,pickledname='qm7b.pkl'):
        pfile = open(pickledname,'r')
        dataset = pickle.load(pfile)

        XX = []
        for i in xrange(len(dataset['X'])):
            d = np.array(dataset['X'][i,:,:])
            first = True
            for dd in d:
                if first:
                    res = dd
                    first = False
                    continue
                res = np.hstack((res,dd))

            XX.append(res)

        XX = np.array(XX)

        dataset['XX'] = XX

        return dataset


def main():
    pickledname = sys.argv[1]
    _qmDL = qmDL()
    dataset = _qmDL.load(pickledname=pickledname)
    print 'Dataset keys',dataset.keys()
    for k in dataset:
        dataset[k] = np.array(dataset[k])
        print '*** Shape',k,  dataset[k].shape



if __name__ == '__main__':
    main()
