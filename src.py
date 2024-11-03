import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

path = kagglehub.dataset_download("arashnic/book-recommendation-dataset")
print("Path to dataset files:", path)


books = pd.read_csv("/Users/jigyasaverma/Desktop/Book Recommendation System/Book Recommendation System/archive/Books.csv", dtype={3: str})
users = pd.read_csv("/Users/jigyasaverma/Desktop/Book Recommendation System/Book Recommendation System/archive/Users.csv")
ratings = pd.read_csv("/Users/jigyasaverma/Desktop/Book Recommendation System/Book Recommendation System/archive/Ratings.csv")

print(ratings.head())
print(books.head())
print(users.head())


#data pre-processing
#finding missing values
ratings.isnull().sum()
users.isnull().sum()
books.isnull().sum()


#handling mising values
user_rating_df = ratings.merge(users, left_on='User-ID', right_on='User-ID')
books['Book-Author'] = books['Book-Author'].fillna('Unknown Author')
books['Publisher'] = books['Publisher'].fillna('Unknown Publisher')
books['Image-URL-L'] = books['Image-URL-L'].fillna('https://default-image-url.com/default.jpg')


print(books.isnull().sum())

#handling duplicate values
print(f"Books Duplicated {books.duplicated().sum()}")
print("\n")
print(f"Ratings Duplicated {ratings.duplicated().sum()}")
print("\n")
print(f"Users Duplicated {users.duplicated().sum()}")


#display info about datasets
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

#performing normailsation for ratings
ratings = ratings[(ratings['Book-Rating'] >= 0) & (ratings['Book-Rating'] <= 10)]
scaler = MinMaxScaler()
ratings['Book-Rating'] = scaler.fit_transform(ratings[['Book-Rating']])


#PERFORMING EDA USING GRAPHS
#Distribution of Book Publication Years
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


#train-tets split model selection
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
train_user_item_matrix = train_data.pivot_table(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)
test_user_item_matrix = test_data.pivot_table(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)
train_user_item_matrix_np = train_user_item_matrix.values
test_user_item_matrix_np = test_user_item_matrix.values
user_similarity = cosine_similarity(train_user_item_matrix_np)


def predict_ratings(user_id, user_item_matrix, user_similarity):
    user_index = user_item_matrix.index.get_loc(user_id)
    user_ratings = user_item_matrix.iloc[user_index]
    weighted_ratings = user_similarity[user_index].dot(user_item_matrix) / np.array(
        [np.abs(user_similarity[user_index]).sum()])
    predicted_ratings = pd.Series(weighted_ratings, index=user_item_matrix.columns)
    return predicted_ratings.sort_values(ascending=False)


#Predict ratings for a specific user in the test set
user_id_example = test_data['User-ID'].iloc[0]
predictions = predict_ratings(user_id_example, train_user_item_matrix, user_similarity)
print(predictions.head(10))

# Function to evaluate model
def evaluate_model(test_data, user_item_matrix, user_similarity):
    predictions = []
    actuals = []

    for user_id in test_data['User-ID'].unique():
        predicted_ratings = predict_ratings(user_id, user_item_matrix, user_similarity)
        user_actual_ratings = test_data[test_data['User-ID'] == user_id]

        for isbn in user_actual_ratings['ISBN']:
            if isbn in predicted_ratings.index:
                predictions.append(predicted_ratings[isbn])
                actuals.append(user_actual_ratings[user_actual_ratings['ISBN'] == isbn]['Book-Rating'].values[0])

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    return rmse

rmse = evaluate_model(test_data, train_user_item_matrix, user_similarity)

print(f'RMSE: {rmse}')

