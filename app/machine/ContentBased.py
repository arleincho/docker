import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
from nltk.stem.snowball import SnowballStemmer
#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

class ContentBased:



	def preprocess(self, creditsStr='../Dataset/tmdb_5000_credits.csv', moviesStr='../Dataset/tmdb_5000_movies.csv', threshold=0.9):

		credits=pd.read_csv(creditsStr)
		movies=pd.read_csv(moviesStr)
		credits.columns = ['id','tittle','cast','crew']
		movies = movies.merge(credits,on='id')

		movies['cast'] = movies['cast'].apply(literal_eval)
		movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
		movies['cast'] = movies['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
		movies['cast'] = movies['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

		movies['crew'] = movies['crew'].apply(literal_eval)
		movies['director'] = movies['crew'].apply(self.get_director)
		movies['director'] = movies['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
		movies['director'] = movies['director'].apply(lambda x: [x,x, x])

		movies['genres'] = movies['genres'].apply(literal_eval)
		movies['genres'] = movies['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
		movies['genres'] = movies['genres'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

		movies['production_companies'] = movies['production_companies'].apply(literal_eval)
		movies['production_companies'] = movies['production_companies'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
		movies['production_companies'] = movies['production_companies'].apply(lambda x: x[:3] if len(x) >=3 else x)
		movies['production_companies'] = movies['production_companies'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

		movies['production_countries'] = movies['production_countries'].apply(literal_eval)
		movies['production_countries'] = movies['production_countries'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
		movies['production_countries'] = movies['production_countries'].apply(lambda x: x[:3] if len(x) >=3 else x)
		movies['production_countries'] = movies['production_countries'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])


		# Keywords has a different treatment, we need to apply some word analysis like stemmer (root of the word) in English:

		movies['keywords'] = movies['keywords'].apply(literal_eval)
		movies['keywords'] = movies['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
		s = movies.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
		s.name = 'keyword'
		s = s.value_counts()
		self.s = s[s > 1]


		stemmer = SnowballStemmer('english')
		movies['keywords'] = movies['keywords'].apply(self.filter_keywords)
		movies['keywords'] = movies['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
		movies['keywords'] = movies['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])


		#Concatenate the keywords, cast, director, genres, production_companies and production_countries
		movies['mix'] = movies['keywords'] + movies['cast'] + movies['director'] + movies['genres'] + movies['production_companies'] + movies['production_countries']

		# Join all values together
		movies['mix'] = movies['mix'].apply(lambda x: ' '.join(x))

		#Concatenate the overview with the tagline and mix
		movies['description'] = movies['overview'].fillna('""') + movies['tagline'].fillna('""') + movies['mix']

		self.movies = movies


		#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
		tfidf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0,stop_words='english')

		#Replace NaN with an empty string
		movies['description'] = movies['description'].fillna('')

		#Construct the required TF-IDF matrix by fitting and transforming the data
		tfidf_matrix = tfidf.fit_transform(movies['description'])

		#Output the shape of tfidf_matrix
		tfidf_matrix.shape

		# Compute the cosine similarity matrix
		self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


		#Construct a reverse map of indices and movie titles
		self.indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()


	


	def get_director(self,x):
	    for i in x:
	        if i['job'] == 'Director':
	            return i['name']
	    return np.nan


	def filter_keywords(self,x):
		words = []
		for i in x:
			if i in self.s:
				words.append(i)
		return words



    # Function that takes in movie title as input and outputs most similar movies
	def get_recommendations(self,title):
	    # Get the index of the movie that matches the title
	    idx = self.indices[title]

	    # Get the pairwsie similarity scores of all movies with that movie
	    sim_scores = list(enumerate(self.cosine_sim[idx]))

	    # Sort the movies based on the similarity scores
	    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
	    
	#     print("Before")
	#     print(sim_scores[0])

	    # Get the scores of the 10 most similar movies
	    sim_scores = sim_scores[1:11]
	#     sim_scores = sim_scores[0:10]
	    
	#     print("After")
	#     print(sim_scores)

	    # Get the movie indices
	    movie_indices = [i[0] for i in sim_scores]

	    # Return the top 10 most similar movies
	    return self.movies['title'].iloc[movie_indices]


