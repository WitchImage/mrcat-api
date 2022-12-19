from django.http.response import HttpResponse
from django.core.handlers.wsgi import WSGIRequest
from django.views.decorators.csrf import csrf_exempt
import sklearn
import tensorflow
import pandas as pd

def index(request):
    versions = f'sklearn version: {sklearn.__version__} - tensorflow version: {tensorflow.__version__}'
    return HttpResponse(versions)

@csrf_exempt
def analyze(request: WSGIRequest) -> HttpResponse:
    class_col, algorithm = request.POST['class_col'], request.POST['algorithm']
    file = request.FILES['file']
    df = pd.read_csv(file)
    print(f'Class: {class_col} - Algorithm: {algorithm}')
    print(df.head())
    return HttpResponse(f'Hola {df.head()}')