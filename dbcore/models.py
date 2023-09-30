from django.db import models


class TrainedModelFile(models.Model):
    file = models.FileField(upload_to='model/')
