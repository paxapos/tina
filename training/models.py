from django.db import models

# Create your models here.
class TrainedProduct(models.Model):
    """
    Trained by human
    """

    product = models.ForeignKey(
        'core.Product',
        on_delete=models.CASCADE,
    )

    score = models.ForeignKey(
        'core.Score',
        on_delete=models.CASCADE,
    )


    picture_path = models.CharField(max_length=200, help_text="File path in FS")
