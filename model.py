import pandas as pd
import numpy as np
from random import sample
from collections import Counter
import ast
import re
import random
import pickle

# model et évaluation
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import roc_auc_score

# notation scientific
np.set_printoptions(suppress=True)
pd.options.display.float_format = '{:.2f}'.format

# data et variable globale
all_users = pd.read_csv("./data/users/all_users.csv")
all_users.drop_duplicates(inplace=True)
all_recipes = pd.read_csv("./data/recipes/all_recipes.csv")
all_recipes.drop_duplicates(inplace=True)
recipe_lookup = all_recipes[["recipe_id","title"]]

# Collaborative filtering for those with at least 3 reviews
# ratings_by_user = all_users.groupby(["user_id","username"])[["rating"]].count().sort_values("rating",ascending=False)
# at_least_3_ids = list(ratings_by_user[ratings_by_user["rating"]>=3].reset_index().user_id)
# users3 = all_users[all_users.user_id.isin(at_least_3_ids)][["user_id","recipe_id","rating"]]
# pickle.dump(users3, open("users3.pkl", "wb"))
users3 = pickle.load(open("./pickle/users3.pkl","rb"))

class utils:
    def __init__(self,all_recipes):
        pass

    def get_category(title):
        df = all_recipes[["category","title"]]
        my_category = df[df.title == title].category
        # Return multiple categories
        # ast.literal turns str rep of list into list
        categories = ast.literal_eval(my_category.values[0])
        return categories

    def recipe_id_to_title(recipe_id):
        df = all_recipes[["recipe_id","title"]]
        my_title = df[df.recipe_id == recipe_id].title
        return my_title.values[0]

    def title_to_id(title):
        df = all_recipes[["recipe_id","title"]]
        my_recipe = df[df.title == title].recipe_id
        return my_recipe.values[0]

    # Preprocessing (lowercasing and stop words)
    def strip_filler(str):
        stop = ["chef", "john's"]
        words = [i for i in str.split() if i.lower() not in stop]
        return " ".join(words)

    # Favorite user's recipes which he has rated more than 4/5
    def known_positives(user_id,threshold=4,new_user=None):
        users = all_users[["user_id","recipe_id","rating"]]
        users = pd.concat([users,pd.DataFrame(new_user)])

        # Join user dataframe and recipes dataframe
        user_preferences = pd.merge(users, recipe_lookup, on='recipe_id', how='left')

        # Get user's reviews rated more than 4
        known_positives = user_preferences[(user_preferences["user_id"] == user_id)&(user_preferences["rating"] >= threshold)]

        known_positives_list = list(known_positives.title)
        return known_positives_list

    # Create a new user from the quizz response
    def create_new_user(quiz_results):
        # Recipes that the user choosed on the quizz
        input = [utils.title_to_id(recipe) for recipe in quiz_results]

        # 5555555 is a random id as the new user will be recreated each time the quizz is filled
        new_user_id = [5555555] * len(input)

        # The 5 recipes
        new_user_recipe_ids = input

        # Ratings 5/5
        new_user_ratings = [5] * len(input)

        # Create new ratings for this new user
        new_user = {'user_id': new_user_id,
        'recipe_id': new_user_recipe_ids,
        'rating': new_user_ratings}

        return new_user

    def count_categories(all_recipes_df):
        all_recipes_df.dropna(axis=0,how='any',inplace=True)
        recipe_categories = all_recipes.drop(["title","category","ingredients"],axis=1)
        categories = []
        # ast.literal turns str rep of list into list
        # dropna otherwise we will experience errors with eval!
        for i in [ast.literal_eval(j) for j in all_recipes_df.category.dropna()] :
            categories.extend(i)
        categories = list(set(categories))
        for category in categories:
            recipe_categories[category] = all_recipes_df["category"].apply(lambda row: int(category in row))

        return recipe_categories.drop(["calories", "ratings", "reviews", "total_mins"],axis=1)

    def similar_to_cat(categories, top_N=10, all_recipes=all_recipes):
        sample_list = []
        matrix = utils.count_categories(all_recipes)
        for category in categories:
            try:
                recipes = list(matrix[matrix[category]==1].recipe_id)
                sample_list.extend(recipes)
            except:
                pass
        return random.sample(sample_list,6)

# Return matrix of ratings (row = users and col = recipes) and user_mapper, recipe_mapper, user_inv_mapper, recipe_inv_mapper
def create_X(df):
    """
    Args:
        df: pandas dataframe containing 3 columns (user_id, recipe_id, rating)

    """
    # Count distict id
    M = df['user_id'].nunique()
    N = df['recipe_id'].nunique()

    user_mapper = dict(zip(np.unique(df["user_id"]), list(range(M)))) # {...,index : (5555555, list(0,1,2,3,4,...,M)), ...}
    recipe_mapper = dict(zip(np.unique(df["recipe_id"]), list(range(N))))

    user_inv_mapper = dict(zip(list(range(M)), np.unique(df["user_id"])))
    recipe_inv_mapper = dict(zip(list(range(N)), np.unique(df["recipe_id"])))

    user_index = [user_mapper[i] for i in df['user_id']]
    item_index = [recipe_mapper[i] for i in df['recipe_id']]

    X = csr_matrix((df["rating"], (user_index, item_index)), shape=(M,N))
    # csr_matrix documentation :
    # row = np.array([0, 0, 1, 2, 2, 2])
    # col = np.array([0, 2, 2, 0, 1, 2])
    # data = np.array([1, 2, 3, 4, 5, 6])

    # for k in 0~5
    # a[row_ind[k], col_ind[k]] = data[k]

    #   0  1  2
    #0 [1, 0, 2]  
    #1 [0, 0, 3]  
    #2 [4, 5, 6]

    return X, user_mapper, recipe_mapper, user_inv_mapper, recipe_inv_mapper

X, user_mapper, recipe_mapper, user_inv_mapper, recipe_inv_mapper = create_X(users3)

# from sklearn.metrics.pairwise import cosine_similarity
# pickle.dump(similarity_matrix, open("similarity_matrix.pkl", "wb"))
# similarity_matrix = cosine_similarity(X,X)

class recommenders:
    def __init__(self):
        pass

    # Top recipes liked by all users (random not sorted)
    def sample_popular(n=24):
        df = all_users[["rating","recipe_id"]].groupby("recipe_id").count().sort_values(by="rating",ascending=False).reset_index()
        top_1000 = [utils.recipe_id_to_title(thing) for thing in df[0:500].recipe_id]
        return sample(top_1000,n)

    def user_user_recommender(top_N, user_id, threshold=4, X_sparse=X, user_mapper=user_mapper, recipe_lookup = recipe_lookup, all_users=all_users,new_user=None):
        similarity_matrix = pickle.load(open("./pickle/similarity_matrix.pkl", "rb"))
        user = user_mapper[user_id]
        # negate for most similar
        similar_users = np.argsort(-similarity_matrix[user])[1:11] # remove original user, pick the top 10 similar users
        #sorted(-similarity_matrix[user])[1:]

        recommended_recipes = []

        # loop through all users to get top_N recipes, only if the recipes > threshold
        for i in similar_users:
            similar_user = (all_users[all_users["user_id"]==user_inv_mapper[i]])
            # Recipes that the similar users rated more than 4/5
            recommended_recipes.extend(list(similar_user[similar_user.rating>=threshold].recipe_id))

        picks = recommended_recipes
        # convert recipe_id to title
        picks = [recipe_lookup.query(f'recipe_id=={i}').title.values[0] for i in picks]

        # remove already tried items
        new_picks = [pick for pick in picks if pick not in utils.known_positives(user_id,threshold,new_user)]

        # remove duplicates & sample 6
        return sample(set(new_picks),6)

    # Search for recommandations after the new user filled the quizz
    def quiz_user_user_recommender(new_user):
        pd_new_user = pd.DataFrame(new_user)
        # concat new_user rows
        new_user_df = pd.concat([users3,pd_new_user])

        # create a X_new
        X_new, user_mapper_new, recipe_mapper_new, user_inv_mapper_new, recipe_inv_mapper_new = create_X(new_user_df)

        return recommenders.user_user_recommender(top_N=30, user_id=5555555, threshold=4, X_sparse=X_new,
        user_mapper=user_mapper_new, recipe_lookup = recipe_lookup, all_users=all_users,new_user=new_user)

    # recipe_categories = utils.count_categories(all_recipes).iloc[:,1:]
    # A = csr_matrix(recipe_categories)
    # del recipe_categories
    # cosine_sim = cosine_similarity(A, A)
    # pickle.dump(cosine_sim, open("cosine_sim.pkl","wb"))

    def item_item_recommender(title, top_N=10, opposite=False, threshold=4, all_recipes=all_recipes, new_user=None, user_id=5555555):
        cosine_sim = pickle.load(open("./pickle/cosine_sim.pkl","rb"))

        recipe_idx = dict(zip(all_recipes['title'], list(all_recipes.index)))
        idx = recipe_idx[title]

        sim_scores = list(enumerate(cosine_sim[idx]))
        if opposite:
            sim_scores.sort(key=lambda x: x[1], reverse=False)
            sim_scores = sim_scores[1:(top_N+1)] # taking the first top_N makes it run a lot faster
            dissimilar_recipes_idx = [i[0] for i in sim_scores]
            picks = list(all_recipes['title'].iloc[dissimilar_recipes_idx])
            new_picks = [pick for pick in picks if pick not in utils.known_positives(user_id,threshold,new_user)]
            return sample(new_picks[0:100],6)

        else:
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:(top_N+1)]
            similar_recipes_idx = [i[0] for i in sim_scores]
            picks = list(all_recipes['title'].iloc[similar_recipes_idx])
            # filter out items chosen, by default filter out new user 5555555
            new_picks = [pick for pick in picks if pick not in utils.known_positives(user_id,threshold,new_user)]
            # choose the top 6 from ranked new_picks to display
            return sample(new_picks[0:10],6)
