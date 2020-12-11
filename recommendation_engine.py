from pprint import pprint
import fasttext
import numpy as np

import time

def fetch_process_articles():
    return {1: ["forest", "fire", "california"], 2: ["coronavirus", "vaccine", "health"], 3: ["politics", "government", "elections"], 4: ["nuclear", "fission", "physics"]}

def recommend(user_preferences, wv):
    article_keywords = fetch_process_articles()

    recommendations = []

    print("User Preferences are: ")
    pprint(user_preferences)

    user_vector = sum(wv[x] for x in user_preferences)

    for article_id, keywords in article_keywords.items():
        article_vector = sum(wv[x] for x in keywords)

        article_score = np.dot(user_vector, article_vector) / (np.linalg.norm(article_vector) * np.linalg.norm(user_vector))

        recommendations.append((article_score, keywords))

    recommendations.sort(reverse=True, key=lambda x: x[0])
    print('Recommendations given are: ')
    pprint(recommendations)


if __name__ == "__main__":
    loaded_model = fasttext.load_model("/home/ag8011/Downloads/FasttextEmbeddingsSimpleEN/wiki.simple.bin")

    recommend(["cancer", "medicine", "politics", "politics"], loaded_model)