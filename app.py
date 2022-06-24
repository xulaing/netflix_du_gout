import pandas as pd
import numpy as np
import random
from flask import request, Flask, render_template
from model import recommenders, utils

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def quiz():
    # choose sample to show for quiz
    most_popular = recommenders.sample_popular()

    if request.method == 'POST':
        quiz_results = request.form.getlist("my_checkbox")

        sample = random.sample(quiz_results, 2)  # for two categories
        title = sample[0]  # title of one of the recipes liked

        ### LES PERSONNES AVEC DES GOÛTS SIMILAIRES ONT AUSSI AIMÉ ! ###
        user_recommended = recommenders.quiz_user_user_recommender(
            utils.create_new_user(quiz_results))

        ### PARCE QUE VOUS AVEZ AIMÉ XXXX ###
        item_recommended = recommenders.item_item_recommender(
            title=title, new_user=utils.create_new_user(quiz_results))

        ### 2 CATEGORIES ###
        # from quiz results, get category, and randomly select 2 to return recipes in that category
        cat1 = utils.get_category(sample[0])
        cat1_recommended = [utils.recipe_id_to_title(
            recipe) for recipe in utils.similar_to_cat(cat1)]

        cat2 = utils.get_category(sample[1])
        cat2_recommended = [utils.recipe_id_to_title(
            recipe) for recipe in utils.similar_to_cat(cat2)]

        cats_recommended = list([cat1_recommended, cat2_recommended])

        ### A DÉCOUVRIR ###
        tastebreaker = recommenders.item_item_recommender(
            title=title, new_user=utils.create_new_user(quiz_results), opposite=True)

        ### LE TOP DU TOP ###
        all = user_recommended + item_recommended
        all = set(all)  # remove duplicates
        # remove recipes user has tried & sample 6
        hybrid_recommended = random.sample([x for x in all if x not in utils.known_positives(
            5555555, new_user=utils.create_new_user(quiz_results))], 6)

        return render_template("result.html",
                               title=title,

                               cats=(list([cat1, cat2]), cats_recommended, [utils.title_to_id(recipe) for recipe in cats_recommended[0]], [
                                     utils.title_to_id(recipe) for recipe in cats_recommended[1]]),
                               # tuple, second element is the image url
                               most_popular=([utils.strip_filler(recipe) for recipe in most_popular],
                                             [utils.title_to_id(recipe) for recipe in most_popular]),

                               quiz_results=([utils.strip_filler(recipe) for recipe in quiz_results],
                                             [utils.title_to_id(recipe) for recipe in quiz_results]),

                               user_recommended=([utils.strip_filler(recipe) for recipe in user_recommended],
                                                 [utils.title_to_id(recipe) for recipe in user_recommended]),

                               item_recommended=([utils.strip_filler(recipe) for recipe in item_recommended],
                                                 [utils.title_to_id(recipe) for recipe in item_recommended], utils.strip_filler(title)),

                               tastebreaker=([utils.strip_filler(recipe) for recipe in tastebreaker],
                                             [utils.title_to_id(recipe) for recipe in tastebreaker], utils.strip_filler(title)),

                               hybrid_recommended=([utils.strip_filler(recipe) for recipe in hybrid_recommended],
                                                   [utils.title_to_id(recipe) for recipe in hybrid_recommended])
                               )

    # landing screen
    return render_template("quiz.html",
                           most_popular=(most_popular, [
                               utils.title_to_id(recipe) for recipe in most_popular])
                           )


if __name__ == '__main__':
    app.run(debug=True)
