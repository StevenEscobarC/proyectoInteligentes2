import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

class ControladorModelo:
    def __int__(self):
        pass

    def entrenar(self, numero):
        if numero == 1:
            return  'Regresión logística',LogisticRegression()
        elif numero == 2:
            return 'KNN',KNeighborsClassifier(n_neighbors=2)
        elif numero == 3:
            return 'MSV',SVC(kernel='linear')
        elif numero == 4:
            return 'Naive Bayes',GaussianNB()
        elif numero == 5:
            return 'Arboles de decisión',DecisionTreeClassifier(max_depth=3)
        elif numero == 6:
            return 'Redes neuronales multicapa',SVC(kernel='linear')
