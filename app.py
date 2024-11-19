import numpy as np
import pandas as pd
from flask import Flask, render_template,request
import pickle



popular_df = pickle.load(open('popular.pkl','rb'))
pt = pickle.load(open('pt.pkl','rb'))
books = pickle.load(open('books.pkl','rb'))
similarity_score = pickle.load(open('similarity_score.pkl','rb'))
# searchdf = pickle.load(open('searchdf.pkl','rb'))

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


# @app.route('/search', methods=['GET'])
# def search():
#     query = request.args.get('query')  # Get the search query
#     if not query:
#         return render_template('index.html', data=[], message="Please enter a search term.")
#
#     # Filter for books with titles matching the query
#     filtered_books = searchdf[searchdf['Book-Title'].str.contains(query, case=False, na=False)]
#
#     if filtered_books.empty:
#         return render_template('index.html', data=[], message="No results found for your search.")
#
#     # Select the first matching book (or all matches if desired)
#     selected_books = filtered_books.head(1)  # Only the first match
#
#     # Extract required fields for display
#     book_name = selected_books['Book-Title'].tolist()
#     author = selected_books['Book-Author'].tolist()
#     image = selected_books['Image-URL-M'].tolist()
#     votes = selected_books['Book-Rating'].tolist()
#     rating = [f"{rating:.1f}" for rating in votes]  # Format ratings
#
#     return render_template('index.html', book_name=book_name, author=author, image=image, votes=votes, rating=rating)


if __name__ =='__main__':
    app.run(debug=True)

