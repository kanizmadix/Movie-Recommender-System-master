from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Load the data into a DataFrame
new_df = pd.read_csv('C:\Users\Kaniz\Pictures\Movie-Recommender-System-master\tmdb_5000_movies.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(new_df['tags'], new_df['title'], test_size=0.2, random_state=42)

# Create a CountVectorizer object
cv = CountVectorizer(max_features=5000, stop_words='english')

# Fit the vectorizer to the training data and transform both the training and testing data
X_train_count = cv.fit_transform(X_train)
X_test_count = cv.transform(X_test)

# Calculate the cosine similarity between the training and testing data
similarity = cosine_similarity(X_test_count, X_train_count)

# Get the indices of the most similar items
indices = np.argsort(-similarity, axis=1)

# Get the actual and predicted labels
actual_labels = y_test.tolist()
predicted_labels = [y_train.iloc[indices[i, 0]] for i in range(len(actual_labels))]

# Calculate the mean average precision (MAP)
def mean_average_precision(actual, predicted):
    precision_sum = 0
    for i in range(len(actual)):
        precision_sum += 1 / (predicted.index(actual[i]) + 1)
    return precision_sum / len(actual)

map_score = mean_average_precision(actual_labels, predicted_labels)
print("Mean Average Precision (MAP):", map_score)

# Calculate the normalized discounted cumulative gain (NDCG)
def ndcg(actual, predicted):
    dcg = 0
    idcg = 0
    for i in range(len(actual)):
        dcg += 1 / np.log2(predicted.index(actual[i]) + 2)
        idcg += 1 / np.log2(i + 2)
    return dcg / idcg

ndcg_score = ndcg(actual_labels, predicted_labels)
print("Normalized Discounted Cumulative Gain (NDCG):", ndcg_score)