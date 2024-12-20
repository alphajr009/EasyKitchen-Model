from django.db import models

class Recipe(models.Model):
    name = models.CharField(max_length=100)
    ingredients = models.TextField()
    recipe_details = models.TextField()

    def __str__(self):
        return self.name
