from flask import Flask, jsonify
import pandas as pd
import numpy as np
from math import sqrt

app = Flask(__name__)
movie = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

final = pd.merge(ratings, movie, on=["movieId"])

final.drop(["genres", "timestamp"], axis=1, inplace=True)


def recommend(id):
    user_df = final[final['userId'] == id]
    Id = final[final['title'].isin(user_df['title'].tolist())]
    userSubsetGroup = Id.groupby(['userId'])
    userSubsetGroup = sorted(userSubsetGroup, key=lambda x: len(x[1]),
                             reverse=True)

    pearsonCorDict = {}
    for name, group in userSubsetGroup:
        # Let's start by sorting the input and current user group so the values aren't mixed up later on
        group = group.sort_values(by='movieId')
        user_df = user_df.sort_values(by='movieId')
        # Get the N for the formula
        n = len(group)
        # Get the review scores for the movies that they both have in common
        temp = user_df[user_df['movieId'].isin(group['movieId'].tolist())]
        # And then store them in a temporary buffer variable in a list format to facilitate future calculations
        tempRatingList = temp['rating'].tolist()
        # put the current user group reviews in a list format
        tempGroupList = group['rating'].tolist()
        # Now let's calculate the pearson correlation between two users, so called, x and y
        Sxx = sum([i ** 2 for i in tempRatingList]) - pow(sum(tempRatingList),
                                                          2) / float(n)
        Syy = sum([i ** 2 for i in tempGroupList]) - pow(sum(tempGroupList),
                                                         2) / float(n)
        Sxy = sum(i * j for i, j in zip(tempRatingList, tempGroupList)) - sum(
            tempRatingList) * sum(tempGroupList) / float(n)

        # If the denominator is different than zero, then divide, else, 0 correlation.
        if Sxx != 0 and Syy != 0:
            pearsonCorDict[name] = Sxy / sqrt(Sxx * Syy)
        else:
            pearsonCorDict[name] = 0
    pearsonDF = pd.DataFrame.from_dict(pearsonCorDict, orient='index')
    pearsonDF.columns = ['similarityIndex']
    pearsonDF['userId'] = pearsonDF.index
    pearsonDF.index = range(len(pearsonDF))
    pearsonDF.head()

    topUsers = pearsonDF.sort_values(by='similarityIndex', ascending=False)[
               0:50]
    topUsers.head()
    topUsersRating = topUsers.merge(ratings, left_on='userId',
                                    right_on='userId', how='inner')
    topUsersRating.drop('timestamp', 1, inplace=True)
    topUsersRating['weightedRating'] = topUsersRating['similarityIndex'] * \
                                       topUsersRating['rating']
    tempTopUsersRating = topUsersRating.groupby('movieId').sum()[
        ['similarityIndex', 'weightedRating']]
    tempTopUsersRating.columns = ['sum_similarityIndex', 'sum_weightedRating']
    recommendation_df = pd.DataFrame()
    # Now we take the weighted average
    recommendation_df['weighted average recommendation score'] = \
    tempTopUsersRating['sum_weightedRating'] / tempTopUsersRating[
        'sum_similarityIndex']
    recommendation_df['movieId'] = tempTopUsersRating.index
    recommendation_df = recommendation_df.sort_values(
        by='weighted average recommendation score', ascending=False)
    final_recommendation = recommendation_df[:8]['movieId'].tolist()[:8]
    return final_recommendation


@app.get('/recommendations/<user_id>')
def get_recommendations(user_id):
    return str(recommend(int(user_id)))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9100, debug=True)
