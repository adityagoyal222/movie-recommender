import streamlit as st
from utils import get_movies,get_movie_id,get_movie_info
from src.utils import recommender


def main():
    st.title("Movie Recommendation System")
    st.write(
        "Choose a movie to get recommendations based on collaborative filtering which"
        "recommends movie based on movies liked by users who watched this movie."
    )
    movie_titles = get_movies()
    movie = st.selectbox("Select a movie:", movie_titles)

    explore_button = st.button("Explore!")
    
    if explore_button:
        recommendations = recommender(movie)
        movie_id_list = get_movie_id(recommendations)
        recommended_movies= get_movie_info(movie_id_list)

        st.subheader("Users who watched {} also liked:".format(movie))

        for rec_movie in recommended_movies:
            st.write(f"**{rec_movie}**")

if __name__ == "__main__":
    main()