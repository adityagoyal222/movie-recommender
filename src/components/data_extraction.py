'''
Data will be imported from artifacts folder and
the movie and user data will be formatted to get
user-item matrix.
'''

import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import extract_movie_name,save_file

import pandas as pd
from sklearn.model_selection import train_test_split

@dataclass
class DataExtractionConfig:
    raw_data_path: str = os.path.join('artifacts','user_item.pkl')
    train_data_path: str = os.path.join('artifacts','train_user_item.pkl')
    test_data_path: str = os.path.join('artifacts','test_user_item.pkl')
    movie_list_path: str = os.path.join('artifacts','movies_list.pkl')

class DataExtraction:
    def __init__(self):
        self.data_extraction_config = DataExtractionConfig()
    
    def initiate_data_extraction(self,movies_path,ratings_path):
        logging.info("Data Extraction has started!")
        try:
            movies_df = pd.read_csv(movies_path)
            ratings_df = pd.read_csv(ratings_path)

            movies_info = movies_df.merge(ratings_df, on='movieId')

            user_rating_threshold = 500
            active_users = movies_info.groupby('userId')['rating'].count() > user_rating_threshold
            active_users_indices = active_users[active_users == True].index

            logging.info("Extracted active users' information")

            movies_watched_by_active_users = movies_info[movies_info['userId'].isin(active_users_indices)]
            movie_rating_threshold = 500
            popular_movies = movies_watched_by_active_users.groupby('movieId')['rating'].count() > movie_rating_threshold
            popular_movie_indices = popular_movies[popular_movies == True].index

            logging.info("Extracted the list of final movies to be used")

            final_list_movies = movies_watched_by_active_users[movies_watched_by_active_users['movieId'].isin(popular_movie_indices)]
            final_list_movies = final_list_movies.drop_duplicates()

            final_movies_info = final_list_movies[['movieId','title']].drop_duplicates()
            save_file(self.data_extraction_config.movie_list_path,final_movies_info)
            
            logging.info("Saved the final list of movie names")

            user_item = final_list_movies.pivot_table(index='title', columns='userId', values='rating', fill_value=0)

            train_user_item,test_user_item = train_test_split(user_item.values, test_size=0.2, random_state=42)

            save_file(self.data_extraction_config.raw_data_path,user_item)
            save_file(self.data_extraction_config.train_data_path,train_user_item)
            save_file(self.data_extraction_config.test_data_path,test_user_item)

            logging.info("Data Extraction has been completed!")

            return(
                train_user_item,
                test_user_item,
                self.data_extraction_config.raw_data_path,
                self.data_extraction_config.train_data_path,
                self.data_extraction_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        