import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
# import pickle

class Demographic:

	# def __init__(self):


	def preprocess(self, creditsStr='/Dataset/tmdb_5000_credits.csv', moviesStr='/Dataset/tmdb_5000_movies.csv', threshold=0.9):

		dir_path = os.path.dirname(os.path.realpath(__file__))
		
		credits=pd.read_csv("%s/%s" % (dir_path, creditsStr))
		movies=pd.read_csv("%s/%s" % (dir_path, moviesStr))
		# movies=pd.read_csv(moviesStr)

		credits.columns = ['id','tittle','cast','crew']
		self.movies = movies.merge(credits,on='id')

		# Mean of vote average
		self.C= movies['vote_average'].mean()

		# Obtain the value of vote count in the 90 percentage of the movies (threshold)
		self.m= movies['vote_count'].quantile(threshold)

		# Creating a new df with the movies greater or equal to the threshold
		self.q_movies = movies.copy().loc[movies['vote_count'] >= self.m]

		# Define a new feature 'score' and calculate its value with `weighted_rating()`
		self.q_movies['score'] = self.q_movies.apply(self.weighted_rating, axis=1)

		# Sort movies based on score calculated above
		self.q_movies = self.q_movies.sort_values('score', ascending=False)




	def weighted_rating(self,x):
		m=self.m 
		C=self.C
		v = x['vote_count']
		R = x['vote_average']
	    # Calculation based on the IMDB formula
		return (v/(v+m) * R) + (m/(m+v) * C)

	def getMostPopular(self):
	    return self.movies.sort_values('popularity', ascending=False)[['id','title','popularity']].head(10)

	def getMostByScored(self):
		return self.q_movies.sort_values('score', ascending=False)[['id','title','score']].head(10)

	def getMostByVoteAverage(self):
	    return self.movies.sort_values('vote_average', ascending=False)[['id','title','vote_average']].head(10)

	def getMostByVoteCount(self):
	    return self.movies.sort_values('vote_count', ascending=False)[['id','title','vote_count']].head(10)

	# def printAll(self):
	# 	print("\nPreprocessed movies (HEAD) \n")
	# 	print(self.q_movies[['title', 'score', 'vote_count', 'vote_average', 'popularity']].head(10))

	# 	print("\nTop 10 Most Popular \n")
	# 	print(self.getMostPopular())

	# 	print("\nTop 10 Weighted Rated \n")
	# 	print(self.getMostByScored())

	# 	print("\nTop 10 Vote Average \n")
	# 	print(self.getMostByVoteAverage())

	# 	print("\nTop 10 Vote Count \n")
	# 	print(self.getMostByVoteCount())


# if __name__ == "__main__": 
#     # print ("Executed when invoked directly")
#     dem = Demographic()
#     dem.preprocess(creditsStr='tmdb_5000_credits.csv', moviesStr='tmdb_5000_movies.csv')
#     dem.printAll()


# else: 
#     # print "Executed when imported"
