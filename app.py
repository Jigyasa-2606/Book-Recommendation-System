import numpy as np
import pandas as pd
from flask import Flask, render_template,request
import pickle

popular_df = pickle.load(open('popular.pkl','rb'))
pt = pickle.load(open('pt.pkl','rb'))
books = pickle.load(open('books.pkl','rb'))
similarity_score = pickle.load(open('similarity_score.pkl','rb'))

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html',
                           book_name = list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_ratings'].values)
                           )
@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books', methods=["post"])
def recommend():
    user_input = request.form.get('user_input')
    index = np.where(pt.index == user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:6]

    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

        data.append(item)

    print(data)

    return render_template('recommend.html',data=data)
@app.route('/about')
def about():
    # Add content for the About page
    return render_template('about.html')


@app.route('/search', methods=['GET'])
def search():
    search_query = request.args.get('query', '').lower()
    print(1)
    books_data = pd.read_csv('/Users/jigyasaverma/Desktop/Book Recommendation System/Book Recommendation System/archive/Books.csv')
    print(2)
    # Filter books based on the search query
    filtered_books = books_data[books_data['Book-Title'].str.contains(search_query, case=False, na=False) |
                                books_data['Book-Author'].str.contains(search_query, case=False, na=False) |
                                books_data['Year-Of-Publication'].astype(str).str.contains(search_query)]

    # Convert filtered books to a list of dictionaries
    books = filtered_books.to_dict(orient='records')
    return render_template('index.html', books=books)


if __name__ =='__main__':
    app.run(debug=True)

