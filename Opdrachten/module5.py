import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
import random

#data inladen
digits = datasets.load_digits()

#arraytjes declareren om van de data te verdelen
TrainingsData = []
TrainingsTarget= []
testData = []
testTarget = []

for i in range(len(digits.data)):
    #staat plus minus, dus deze random methode versta ik er onder,
    #ben benieuwed als er ook andere methodes zijn.
    if (random.choice([1,1,0] )):
        TrainingsData.append(digits.data[i])
        TrainingsTarget.append(digits.target[i])
    else:
        testData.append(digits.data[i])
        testTarget.append(digits.target[i])

print(len(TrainingsData))
print(len(testData))

clf = svm.SVC(gamma=0.001, C=100)
X,y = TrainingsData, TrainingsTarget
clf.fit(X,y)

Truecounter = 0
predictestTest = clf.predict(testData)
for i in range(len(testData)):
    if( predictestTest[i] == testTarget[i]):
        Truecounter += 1

print(Truecounter / len(testData))

print(clf.predict(digits.data[-4:-3]))
plt.imshow(digits.images[-4], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()