from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
print(len(iris.data))
n_samples = int(len(iris) * 0.8)
trainData = iris.data[:n_samples]
testData = iris.data[n_samples:]
trainTarget = iris.target[:n_samples]
testTarget = iris.target[n_samples:]

gnb = GaussianNB()
gnb.fit(trainData, trainTarget)
predictTarget = gnb.predict(testData)

print("Number of mislabeled points out of a total %d points : %d"
      %(iris.data.shape[0],(testTarget != predictTarget).sum()))

