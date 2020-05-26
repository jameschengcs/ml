import numpy as np
from sklearn import datasets, svm

boston = datasets.load_boston()
Xtrain = boston.data[:-20, :]
Xtest = boston.data[-20:, :]
Ytrain = boston.target[:-20]
Ytest = boston.target[-20:]

models = [svm.SVR(kernel='linear'),
          svm.SVR(kernel='rbf', C=1e3, gamma=0.1)]

for svr in models:
    svr.fit(Xtrain, Ytrain)

for svr in models:
    Ypred = svr.predict(Xtest)
    print(Ytest, Ypred)
