# Generated by Django 4.0.5 on 2022-06-17 03:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('training', '0005_alter_trainedproduct_time_date'),
    ]

    operations = [
        migrations.AlterField(
            model_name='trainedproduct',
            name='time_date',
            field=models.DateTimeField(auto_now_add=True, verbose_name='Created'),
        ),
    ]
