import random

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

books = pd.read_csv("/Users/jigyasaverma/Desktop/Book Recommendation System/Book Recommendation System/archive/Books.csv", dtype=str)
users = pd.read_csv("/Users/jigyasaverma/Desktop/Book Recommendation System/Book Recommendation System/archive/Users.csv",nrows=100000)
ratings = pd.read_csv("/Users/jigyasaverma/Desktop/Book Recommendation System/Book Recommendation System/archive/Ratings.csv")


#POPULARITY BASED RECOMMENDATION SYSTEM

ratings_with_name = ratings.merge(books, on='ISBN')
num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating':'num_ratings'}, inplace= True)

print(num_rating_df)
print(ratings_with_name.dtypes)
ratings_with_name['Book-Rating'] = pd.to_numeric(ratings_with_name['Book-Rating'], errors='coerce')
try:
  avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
  avg_rating_df.rename(columns={'Book-Rating':'avg_ratings'}, inplace= True)
  print(avg_rating_df)
except TypeError as e:
  print("Error encountered while calculating mean: ", e)

popularity_df = num_rating_df.merge(avg_rating_df,on='Book-Title')
print(popularity_df)
popular_df = popularity_df[popularity_df['num_ratings']>=250].sort_values('avg_ratings',ascending=False).head(50)
popular_df = popular_df.merge(books,on='Book-Title')
popular_df = popular_df[['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_ratings']]




#by collaborative filtering

x = ratings_with_name.groupby('User-ID').count()['Book-Rating']>200
users_highly_rated = x[x].index
filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(users_highly_rated)]
y= filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books = y[y].index
final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

pt = final_ratings.pivot_table(index ='Book-Title',columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)
print(pt)
similarity_score = cosine_similarity(pt)



def recommend(book_name):
  # index fetch
  index = np.where(pt.index == book_name)[0][0]
  similar_items = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:5]

  data = []
  for i in similar_items:
    item = []
    temp_df = books[books['Book-Title'] == pt.index[i[0]]]
    item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
    item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
    item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

    data.append(item)

  return data

recommend('Animal Farm')


