from django.db import models

class Product(models.Model):
    """
    Product model class
    """

    name = models.CharField(max_length=20, help_text="Enter product name")
    alias = models.CharField(max_length=20, help_text="Enter product alias", unique=True)

class Score(models.Model):
    """
    Score class to store score number for products
    """
    number = models.IntegerField( help_text="Enter product score", unique=True)
    description = models.CharField(max_length=80, help_text="Enter product alias")
