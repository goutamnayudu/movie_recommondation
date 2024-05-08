import numpy as np
import pandas as pd
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# loading data from csv file to pandas dataframe
movies_data = pd.read_csv('C:\\Users\\Visitor\\Desktop\\Movie Recommndation\\movies.csv')

# Selecting the relevant features for recommendation
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# Data Cleaning
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_data = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data[
    'cast'] + ' ' + movies_data['director']

# converting Categorical data to Numeric data (Feature Vectors)
feature_vectors = TfidfVectorizer().fit_transform(combined_data)

# getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)

# create a CSV file to store sorted similarity scores
csv_filename = 'sorted_similarity_scores.csv'

# write the header to the CSV file
header = ['Movie Title'] + movies_data['title'].tolist()
with open(csv_filename, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)

# write the sorted similarity scores to the CSV file row by row
for i in range(len(similarity)):
    # getting a list of similar movies
    similarity_score = list(enumerate(similarity[i]))

    # sorting the movies based on their similarity score
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    # add the sorted similarity scores to the CSV file row by row
    row_data = [movies_data.at[i, 'title']] + [round(similarity_percentage * 100, 2) for _, similarity_percentage in sorted_similar_movies]
    with open(csv_filename, 'a', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(row_data)

print(f'Sorted similarity scores written to {csv_filename}')
