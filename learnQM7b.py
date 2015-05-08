from sklearn.cross_validation import train_test_split
from qmDataLoader import qmDL
import sys
from sklearn.linear_model import MultiTaskLasso
from sklearn.metrics import mean_absolute_error
from sklearn.kernel_ridge import KernelRidge
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR

class multiTargetRegressor(object):
    def __init__(self,rObject=None):
        self.R = rObject
        self.RList = list()

    def fit(self,X,Y):
        self.RList = list()
        self.N = len(Y[0])
        for y in xrange(self.N):
            self.RList.append(self.R.fit(X,Y[:,y]) )
        return self

    def predict(self,X):
        predictions = self.RList[0].predict(X)
        for i in xrange(1,self.N):
            print predictions.shape
            predictions = np.vstack((predictions, self.RList[i].predict(X) ))

        return np.transpose(predictions)

def main():
    pickledname = sys.argv[1]
    _qmDL = qmDL()
    dataset = _qmDL.load(pickledname=pickledname)

    X, Y , labels = dataset['XX'], dataset['T'], dataset['names']

    #5000 training samples, with 2211 test samples
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=2211, random_state=42)
    print 'Len X train , test:', len(X_train), len(X_test)


    regressor = MultiTaskLasso().fit(X_train,Y_train)
    #r = SVR()
    #regressor = multiTargetRegressor(rObject=r).fit(X_train,Y_train)
    Y_pred = regressor.predict(X_test)

    print Y_pred
    print 'Y_pred', Y_pred.shape

    for i in xrange(len(labels)):
        print '*** MAE ', labels[i],
        print mean_absolute_error(Y_test[:,i], Y_pred[:,i])


if __name__ == '__main__':
    main()
