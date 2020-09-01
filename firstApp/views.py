from django.shortcuts import render

from django.core.files.storage import FileSystemStorage

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow import Graph

import json

img_height, img_width = 100, 100
with open('D:/s/jsonlabeling','r') as f:
    labelInfo = f.read()
labelInfo = json.loads(labelInfo)

model_graph = Graph()
with model_graph.as_default():
    tf_session=tf.compat.v1.Session()
    with tf_session.as_default():
        model = load_model('D:/s/한국인안면/modelsave.h5')

# Create your views here.

def index(request):
    context = {'a':1}
    return render(request,'index.html',context)

def predictImage(request):
    print(request)
    print(request.POST.dict())
    fileObj=request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(filePathName)
    testimage = '.' + filePathName

    img=image.load_img(testimage,target_size=(img_height,img_width))
    x = image.img_to_array(img)
    x = x.reshape(1,img_height,img_width,3)
    with model_graph.as_default():
        with tf_session.as_default():
            predi=model.predict(x)




    import numpy as np

    # predictedlabel=labelInfo ('{}'.format(np.argmax(predi)))

    context={'filePathName' : filePathName, 'predictedLabel' : predi}

    return render(request, 'index.html', context)