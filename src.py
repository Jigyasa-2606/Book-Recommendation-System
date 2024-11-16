import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
path = kagglehub.dataset_download("arashnic/book-recommendation-dataset")
print("Path to dataset files:", path)


books = pd.read_csv("/Users/jigyasaverma/Desktop/Book Recommendation System/Book Recommendation System/archive/Books.csv", dtype=str)
users = pd.read_csv("/Users/jigyasaverma/Desktop/Book Recommendation System/Book Recommendation System/archive/Users.csv",nrows=100000)
ratings = pd.read_csv("/Users/jigyasaverma/Desktop/Book Recommendation System/Book Recommendation System/archive/Ratings.csv")
num_rows = len(books)
num_rows_users = len(users)
num_rows_ratings = len(ratings)

# print(num_rows)
# print(num_rows_users)
# print(num_rows_ratings)
#
# print(books.head())
# print(users.head())
# print(ratings.head())

#data pre-processing
#finding missing values
ratings.isnull().sum()
users.isnull().sum()
books.isnull().sum()


#handling mising values

books['Book-Author'] = books['Book-Author'].fillna('Unknown Author')
books['Publisher'] = books['Publisher'].fillna('Unknown Publisher')
books['Image-URL-L'] = books['Image-URL-L'].fillna('https://default-image-url.com/default.jpg')

users['Age'].fillna(users['Age'].mean(), inplace=True)
print(users.isnull().sum())

#handling duplicate values
print(f"Books Duplicated {books.duplicated().sum()}")
print("\n")
print(f"Ratings Duplicated {ratings.duplicated().sum()}")
print("\n")
print(f"Users Duplicated {users.duplicated().sum()}")

# display info about datasets
print(books.info())
print(users.info())
print(ratings.info())

#staistical approach of data
print(books.describe())
print(users.describe())
print(ratings.describe())

#shape of dataset
print(books.shape)
print(users.shape)
print(ratings.shape)

#converting type from string to int
books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'], errors='coerce').fillna(0).astype(int)
#
# performing normailsation for ratings
ratings = ratings[(ratings['Book-Rating'] >= 0) & (ratings['Book-Rating'] <= 10)]
scaler = MinMaxScaler()
ratings['Book-Rating'] = scaler.fit_transform(ratings[['Book-Rating']])

#
# PERFORMING EDA USING GRAPHS
# Distribution of Book Publication Years
plt.figure(figsize=(12, 6))
sns.histplot(books['Year-Of-Publication'], bins=30, kde=True)
plt.title('Distribution of Book Publication Years')
plt.xlabel('Publication Year')
plt.ylabel('Frequency')
plt.show()

#most common authors
plt.figure(figsize=(12, 6))
books['Book-Author'].value_counts().nlargest(10).plot(kind='bar')
plt.title('Top 10 Most Common Authors')
plt.xlabel('Author')
plt.ylabel('Number of Books')
plt.xticks(rotation=45)
plt.show()


#number of books publishes over time
plt.figure(figsize=(12, 6))
books['Year-Of-Publication'].value_counts().sort_index().plot(kind='line')
plt.title('Number of Books Published Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Books')
plt.xticks(rotation=45)
plt.show()

#now for distribution of books by users age
plt.figure(figsize=(12, 6))
sns.histplot(users['Age'], bins=20, kde=True)
plt.title('Distribution of User Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Count of Users by Location
plt.figure(figsize=(12, 6))
users['Location'].value_counts().nlargest(10).plot(kind='bar')
plt.title('Top 10 Locations by Number of Users')
plt.xlabel('Location')
plt.ylabel('Number of Users')
plt.xticks(rotation=45)
plt.show()

#age group analysis
bins = [0, 18, 25, 35, 45, 55, 65, 100]
labels = ['0-18', '19-25', '26-35', '36-45', '46-55', '56-65', '66+']
users['Age Group'] = pd.cut(users['Age'], bins=bins, labels=labels, right=False)

plt.figure(figsize=(12, 6))
users['Age Group'].value_counts().sort_index().plot(kind='bar')
plt.title('Number of Users by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Number of Users')
plt.show()

#distribution of book ratoings
plt.figure(figsize=(12, 6))
sns.histplot(ratings['Book-Rating'], bins=5, kde=False)
plt.title('Distribution of Book Ratings')
plt.xlabel('Book Rating')
plt.ylabel('Frequency')
plt.xticks([1, 2, 3, 4, 5])
plt.show()

#Count of Ratings per Use
plt.figure(figsize=(12, 6))
ratings['User-ID'].value_counts().nlargest(10).plot(kind='bar')
plt.title('Top 10 Users by Number of Ratings')
plt.xlabel('User ID')
plt.ylabel('Number of Ratings')
plt.xticks(rotation=45)
plt.show()


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



import pickle
pickle.dump(popular_df,open('popular.pkl','wb'))

pickle.dump(pt,open('pt.pkl','wb'))
pickle.dump(books,open('books.pkl','wb'))
pickle.dump(similarity_score,open('similarity_score.pkl','wb'))

print(popular_df.columns)
