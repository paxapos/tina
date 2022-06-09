from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from core.models import Product


def train(request):

    product_list = Product.objects.order_by('-name')[:5]

    template = loader.get_template('training/train.html')
    context = {
        'product_list': product_list,
    }
    return HttpResponse(template.render(context, request))
