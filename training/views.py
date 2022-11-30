import json
import os
from uuid import uuid4
from datetime import datetime

from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.template import loader
from django.http import (
    HttpRequest, HttpResponse,
    HttpResponseBadRequest, 
    HttpResponseServerError, 
    JsonResponse
)


from tina.settings import BASE_DIR, MEDIA_ROOT
from ia.ia_engine import IaEngine
from core.models import Product, Score
from training.models import TrainedProduct, TrainedProductPicture
from training.camera import take_pictures


def train_admin(request) -> HttpResponse:
    product_list = Product.objects.order_by('name')
    score_list = Score.objects.order_by('number')
    template = loader.get_template('training/train.html')
    context = {
        'product_list': product_list,
        'score_list': score_list
    }
    return HttpResponse(template.render(context, request))


def capture(request: HttpRequest) -> HttpResponse:
    # params = json.loads(request.body)
    qty = int(request.GET.get('qty', 1))
    delay = int(request.GET.get('delay', 0))
    pics = take_pictures(BASE_DIR / 'training/static/temp', 
                         qty=qty,
                         delay=delay)
    data = {'pics': pics}
    return JsonResponse(data)


def upload(request: HttpRequest) -> HttpResponse:
    params = json.loads(request.body)
    product_alias = params.get('productAlias', None)
    score_number = params.get('scoreNumber', None)
    file_names = params.get('fileNames', [])

    if not product_alias or not score_number or not file_names:
        return HttpResponseBadRequest(
            json.dumps(
                {'error':'One or more params missing'}
                ))

    model = TrainedProduct(
        product = Product.objects.get(alias=product_alias),
        score = Score.objects.get(number=score_number),
        date_time = datetime.now()
        )
    model.save()
    product_pk = model.pk

    for file_name in file_names:
        uuid = uuid4().__str__()
        path = f'{product_alias}/{score_number}/'
        name = f'{uuid[19:]}.jpg'
        file_path = path + name

        trained_product_picture = TrainedProductPicture(
            trained_product=TrainedProduct.objects.get(pk=product_pk),
                            picture_path=file_path)

        old_file = BASE_DIR / f'training/static/temp/{file_name}'
        new_path = BASE_DIR / f'training/pics/{path}'
        new_file = new_path / name

        try:
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            os.rename(old_file, new_file)

        except Exception as e:
            print(e)
            return HttpResponseServerError(
                json.dumps(
                    {'error':'The pictures could not be saved'}
                    ))

        trained_product_picture.save()

    return JsonResponse({'picture_path': str(new_file)})

def remove(request) -> HttpResponse:
    params = json.loads(request.body)
    file_names = params.get('fileNames', None)
    if not file_names:
        return HttpResponseBadRequest()
    for file_name in file_names:
        os.remove(BASE_DIR / f'training/static/temp/{file_name}')
    return JsonResponse({'deleted': file_names})


def train(request) -> HttpResponse:
    IaEngine.train()
    return HttpResponse("")

def training(request):
    return render(request, "training/training.html")

def pictures(request):
    return render(request, "training/pictures.html")

def predict(request):
    if request.method == "POST":
        uploaded_picture = request.FILES["picture"]
        fs = FileSystemStorage()
        fs.delete(uploaded_picture.name)
        fs.save(uploaded_picture.name, uploaded_picture)
        picture_path = os.path.join(MEDIA_ROOT + '/' + uploaded_picture.name)
        product = request.POST['product']
        prediction = IaEngine.predict(product, product, picture_path)
        print(prediction)  
        return render(request, "predict/predict.html", {'prediction': prediction})
    else:
        return render(request, "predict/predict.html")

