import os
import pandas as pd
import requests
from src.utils import load_file
import streamlit as st

def get_movie_name(movie_id):
    movies_df = pd.read_csv(os.path.join('artifacts', 'movies.csv'))
    title_series = movies_df[movies_df['movieId'] == movie_id]['title']
    
    if not title_series.empty:
        title = title_series.values[0]
        return title
    else:
        st.warning(f"Movie ID {movie_id} not found.")
        return "Unknown"

def get_movies():
    user_item = load_file(os.path.join('artifacts','user_item.pkl'))
    return list(user_item.index)

def get_movie_id(movie_names): 
    movies_df = pd.read_csv(os.path.join('artifacts','movies.csv'))
    movie_id_list = movies_df[movies_df['title'].isin(movie_names)]['movieId'].to_list()

    return movie_id_list

def get_movie_info(movie_id_list):
    movie_names = []

    for recommended_movie_id in movie_id_list:
        movie_name = get_movie_name(recommended_movie_id)
        movie_names.append(movie_name)

    return movie_names