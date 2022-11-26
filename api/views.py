from django.http.response import HttpResponse
import sklearn
import tensorflow

def index(request):
    versions = f'sklearn version: {sklearn.__version__} - tensorflow version: {tensorflow.__version__}'
    return HttpResponse(versions)