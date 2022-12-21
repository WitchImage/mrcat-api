import graphviz
import numpy as np
from sys import argv
from typing import Any, Literal
from sklearn import tree
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_boston, load_breast_cancer, load_iris, load_wine
import pandas as pd

def analyze(dataset: pd.DataFrame, algorithm: str):
    if (algorithm == 'kmeans'):
        kmeans(dataset, 2)
    elif (algorithm == 'knn'):
        knn(dataset, 2)

def id3(dataset: Any):
    X, y = dataset.data, dataset.target
    clf = tree.DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(X, y)

    dot_data: Any = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data, format="jpg")
    graph.render("modelo")
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=dataset.feature_names,
                                    class_names=dataset.target_names,
                                    filled=True, rounded=True,
                                    special_characters=True,)
    graph = graphviz.Source(dot_data)
    graph

def kmeans(dataset: Any, k: int):
    X = np.array([[1, 2], [1, 4], [1, 0],
                [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(n_clusters=k, random_state=0).fit(dataset)
    kmeans.labels_

    cluster_indices = kmeans.predict([[0, 0], [12, 3]])
    return kmeans.cluster_centers_
    print("Cluster centers:", kmeans.cluster_centers_)

def nbayes(dataset: Any):
    X, y = dataset.data, dataset.target
    X_test: Any
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    return ("Number of mislabeled points out of a total %d points : %d"
        % (X_test.shape[0], (y_test != y_pred).sum()))
    print("Accuracy:", gnb.score(X_test, y_test))

def knn(dataset: Any, k: int, distance: Literal["manhattan", "euclidean"] = "euclidean"):
    distances = {"manhattan": 1, "euclidean": 2}
    d = distances[distance]
    X, y = dataset.data, dataset.target
    neigh = KNeighborsClassifier(n_neighbors=k, metric=d)
    neigh.fit(X, y)
    return(neigh.predict([[1, 1]]))

def svm(dataset: Any):
    X, y = dataset.data, dataset.target
    clf = SVC()
    clf.fit(X, y)
    print(clf.predict([[1, 1]]))
    return ("support vectors: ", clf.support_vectors_)
    print("number of supports vectors for each class: ", clf.n_support_)

def load_dataset(dataset: Literal["boston", "iris", "cancer", "wine"]):
    datasets = {
        "boston": load_boston,
        "iris": load_iris,
        "cancer": load_breast_cancer,
        "wine": load_wine,
    }
    # return datasets[dataset](return_X_y=True)
    return datasets[dataset]()

def call_method(method: str, dataset: pd.DataFrame):
    if method == "id3": return id3(dataset)
    elif method == "kmeans": return kmeans(dataset)
    elif method == "nbayes": return nbayes(dataset)
    elif method == "knn": return knn(dataset)
    elif method == "svm": return svm(dataset)
