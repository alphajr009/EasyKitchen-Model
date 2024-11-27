# recipes/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from sklearn.feature_extraction.text import TfidfVectorizer
from .models import Recipe
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RecipePredictionView(APIView):
    def post(self, request, *args, **kwargs):
        ingredients = request.data.get('ingredients', [])

        if not ingredients:
            return Response({"error": "Ingredients list cannot be empty."}, status=400)

        ingredients_text = ', '.join(ingredients)

        if not ingredients_text.strip():  # Check if ingredients are just stopwords or empty
            return Response({"error": "Ingredients list contains no valid terms."}, status=400)

        # Get recipe data from the database
        recipes = Recipe.objects.all()

        # Convert the queryset to a list to allow indexing
        recipes_list = list(recipes)

        # Prepare the recipe data and labels
        recipe_data = []
        recipe_labels = []
        for recipe in recipes_list:
            recipe_data.append(recipe.ingredients)
            recipe_labels.append(recipe.name)

        # Initialize the vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')

        try:
            # Fit and transform the recipe data
            recipe_matrix = vectorizer.fit_transform(recipe_data)
            input_vector = vectorizer.transform([ingredients_text])

            # Calculate cosine similarity
            similarities = cosine_similarity(input_vector, recipe_matrix)

            # Find the most similar recipe
            best_match_idx = np.argmax(similarities)
            best_recipe = recipes_list[best_match_idx]  # Access the list instead of queryset

            return Response({
                'recipe_name': best_recipe.name,
                'recipe_details': best_recipe.recipe_details
            })
        except ValueError as e:
            return Response({"error": str(e)}, status=500)
