from django.http.response import HttpResponse, JsonResponse
from django.core.handlers.wsgi import WSGIRequest
from django.views.decorators.csrf import csrf_exempt
import sklearn
import tensorflow
import pandas as pd
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
import json
# import pandas as pd

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

    # cluster_indices = kmeans.predict([[0, 0], [12, 3]])
    return kmeans.cluster_centers_
    # print("Cluster centers:", kmeans.cluster_centers_)

def nbayes(dataset: pd.DataFrame, class_col: str):
    X, y = dataset.drop(class_col, axis=1), dataset.iloc[:, -1:]
    print(X)
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    return f"Accuracy: {gnb.score(X_test, y_test)}"
    return ("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
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
    return f"support vectors: {clf.support_vectors_}"
    # print("number of supports vectors for each class: ", clf.n_support_)

def load_dataset(dataset: Literal["boston", "iris", "cancer", "wine"]):
    datasets = {
        "boston": load_boston,
        "iris": load_iris,
        "cancer": load_breast_cancer,
        "wine": load_wine,
    }
    # return datasets[dataset](return_X_y=True)
    return datasets[dataset]()

def call_method(method: str, dataset: pd.DataFrame, class_col: str):
    dataset.reset_index(drop=True, inplace=True)
    if method == "id3": return id3(dataset)
    elif method == "kmeans": return kmeans(dataset, 2)
    elif method == "nbayes": return nbayes(dataset, class_col)
    elif method == "knn": return knn(dataset, 2)
    elif method == "svm": return svm(dataset)

def index(request):
    versions = f'sklearn version: {sklearn.__version__} - tensorflow version: {tensorflow.__version__}'
    return HttpResponse(versions)

@csrf_exempt
def analyze(request: WSGIRequest) -> JsonResponse:
    class_col, algorithm = request.POST.get('class_col', ''), request.POST.get('algorithm', '')
    file = request.FILES['file']
    df = pd.read_csv(file)
    print(f'Class: {class_col} - Algorithm: {algorithm}')
    print(df.head())
    res = call_method(algorithm, df, class_col)
    print('Response', res)

    response = json.loads(res)
    return JsonResponse(response, status=200, safe=False)