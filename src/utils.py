import os
import sys
import re
import dill
import numpy as np

from src.exception import CustomException
from src.exception import CustomException
from src.model.model_definition import DeepAutoEncoder
from src.model.utils import loss_function

from tensorflow.keras.optimizers import Adam
import requests

def extract_movie_name(title):
    try:
        match = re.match(r'^(.*?)\s*\(\d{4}\)?$', title)
        return match.group(1) if match else title
    except Exception as e:
        CustomException(e, sys)

def save_file(file_path, file):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(file, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_file(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(model_file_path,embeddings_file_path,train_data,test_data):
    
    try:
        input_shape = train_data.shape[1]
        layers = [256,256,512,256,256]
        activation = 'selu'
        output_activation = 'selu'
        dropout = 0.8
        regularization_encoder = regularization_decoder = 0.001

        model_definition = DeepAutoEncoder(input_shape,layers,activation,output_activation,dropout,regularization_encoder,regularization_decoder)
        model = model_definition.get_model()

        model.compile(optimizer=Adam(learning_rate=0.001),loss=loss_function)

        model.fit(x=train_data,y=train_data,epochs=100,batch_size=64,validation_split=0.2)

        loss = model.evaluate(test_data,test_data)

        encoder_model = model_definition.get_encoder_model()

        user_item = load_file(os.path.join('artifacts','user_item.pkl'))
        movie_embeddings = encoder_model.predict(user_item.values)

        save_file(model_file_path, encoder_model)
        save_file(embeddings_file_path, movie_embeddings)

        return loss
        
    except Exception as e:
        raise CustomException(e,sys)
    
def recommender(movie):
    try:
        user_item = load_file(os.path.join('artifacts','user_item.pkl'))
        movie_embeddings = load_file(os.path.join('artifacts','embeddings.pkl'))
        
        movie_index = user_item.index.get_loc(movie)
        movie_embedding = movie_embeddings[movie_index]

        norm_1 = np.linalg.norm(movie_embeddings, axis = 1)
        norm_2 = np.linalg.norm(movie_embedding)
        cosine_similarity = np.dot(movie_embeddings, movie_embedding)/(norm_1*norm_2)

        similar_movie_indices = np.argsort(cosine_similarity)[::-1]
        similar_movie_indices = [movie_indices for movie_indices in similar_movie_indices if movie_indices != movie_index]

        top = 4
        top_movies = similar_movie_indices[:top]
        top_movie_names = user_item.index[top_movies].tolist()

        return top_movie_names

    except Exception as e:
        raise CustomException(e,sys)