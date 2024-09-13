import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import networkx as nx

# Load datasets
movies_df = pd.read_csv('tmdb_5000_movies.csv')
credits_df = pd.read_csv('tmdb_5000_credits.csv')

# Merge datasets
movies_df = movies_df.merge(credits_df, left_on='id', right_on='movie_id', how='left')

# 1. Genre Distribution Plot
movies_df['genres'] = movies_df['genres'].apply(lambda x: [genre['name'] for genre in eval(x)])
all_genres = [genre for sublist in movies_df['genres'].tolist() for genre in sublist]
genre_counts = pd.Series(all_genres).value_counts()

plt.figure(figsize=(12, 6))
sns.barplot(x=genre_counts.index, y=genre_counts.values, palette="viridis")
plt.title('Genre Distribution')
plt.ylabel('Number of Movies')
plt.xlabel('Genre')
plt.xticks(rotation=45)
plt.show()

# 2. Budget vs Revenue Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(movies_df['budget'], movies_df['revenue'], alpha=0.5)
plt.title('Budget vs Revenue')
plt.xlabel('Budget ($)')
plt.ylabel('Revenue ($)')
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.show()

# 3. Popularity Trend Line
movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')
movies_df['release_year'] = movies_df['release_date'].dt.year
popularity_trend = movies_df.groupby('release_year')['popularity'].mean()

plt.figure(figsize=(10, 6))
plt.plot(popularity_trend.index, popularity_trend.values, marker='o', linestyle='-')
plt.title('Popularity Trend Over Time')
plt.xlabel('Year')
plt.ylabel('Average Popularity')
plt.grid(True)
plt.show()

# 4. Word Cloud for Keywords
movies_df['keywords'] = movies_df['keywords'].apply(lambda x: ' '.join([keyword['name'] for keyword in eval(x)]))
all_keywords = ' '.join(movies_df['keywords'])

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_keywords)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Keywords')
plt.show()

# 5. Actor and Director Network
# Extract top actors and directors
movies_df['cast'] = movies_df['cast'].apply(lambda x: [member['name'] for member in eval(x)][:3])
movies_df['director'] = movies_df['crew'].apply(lambda x: [member['name'] for member in eval(x) if member['job'] == 'Director'])

# Create edge list for the network
edges = []
for _, row in movies_df.iterrows():
    for actor in row['cast']:
        for director in row['director']:
            edges.append((actor, director))

# Build the network graph
G = nx.Graph()
G.add_edges_from(edges)

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.15, iterations=20)
nx.draw(G, pos, with_labels=True, node_size=20, font_size=10, font_color='black', edge_color='gray')
plt.title('Actor and Director Network')
plt.show()
