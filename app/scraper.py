import fasttext
import json
import os
import pandas as pd
import random
import requests
import time

from bs4 import BeautifulSoup
from datetime import datetime
from langdetect import detect
from preprocessing import preprocessing_input
from utils import DATA_DIR, MODEL_DIR

fasttext.FastText.eprint = lambda x: None


def save_as_csv(csv_file):
    today_date = datetime.now().date()
    filename = "data_" + str(today_date) + ".csv"
    csv_file.to_csv(os.path.join(DATA_DIR, filename), index=False)
    # with open(os.path.join(DATA_DIR, filename), "w", encoding="utf-8") as f:
    #     json.dump(json_file, f)


def scrape_webpage(temp_url):
    review_titles = []
    #star_ratings = []
    review_bodies = []
    votes = []
    verifications = []
    page_num = 1
    flag = 0

    while flag == 0 and page_num <= 1000:
        try:
            url = temp_url + "&pageNumber=" + str(page_num)
            # print(url)
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
                "Accept-Language": 'en-US'
            }

            r = requests.get(url, headers=headers)
            soup = BeautifulSoup(r.content, 'html.parser')
            reviews = soup.find_all('div', {'data-hook': 'review'})

            for item in reviews:
                review_title = item.find(
                    'a', {'data-hook': 'review-title'}).text
                review_body = item.find(
                    'span', {'data-hook': 'review-body'}).text.strip()
                verified = item.find(
                    'span', {'data-hook': 'avp-badge'}).text.strip()
                if verified == 'Verified Purchase':
                    verifications.append(True)
                else:
                    verifications.append(False)

                review_titles.append(review_title)
                review_bodies.append(review_body)
            print(page_num)
            page_num += 1

        except:
            print("No more comments, now saving the scraped data")
            print(os.path.join(MODEL_DIR, 'lid.176.ftz'))
            print("HI")
            flag = 1

    # Not checking whether its English when scraping cause it eventually cause the scraping terminates earlier
    print(os.path.join(MODEL_DIR, 'lid.176.ftz'))
    lang_detector = fasttext.load_model(os.path.join(MODEL_DIR, 'lid.176.ftz'))
    titles_to_remove = []
    bodies_to_remove = []

    #i = 0
    for i in range(len(review_titles)):
        review_title_lang = lang_detector.predict(
            review_titles[i].strip(), k=3)
        if review_title_lang[0][0] != '__label__en':
            if review_title_lang[0][1] == '__label__en':
                if review_title_lang[1][1] < 0.158:
                    titles_to_remove.append(i)
                    bodies_to_remove.append(i)
                    verifications[i] = "None"
                    print(review_title_lang)
                    print(review_titles[i])
            else:
                titles_to_remove.append(i)
                bodies_to_remove.append(i)
                verifications[i] = "None"
                print(review_title_lang)
                print(review_titles[i])
        # i+=1

    review_titles = [review_titles[i] for i in range(
        len(review_titles)) if i not in titles_to_remove]
    review_bodies = [review_bodies[i] for i in range(
        len(review_bodies)) if i not in bodies_to_remove]
    verifications = [elem for elem in verifications if elem != "None"]
    print(len(review_titles))
    print(len(review_bodies))
    print(len(verifications))
    if len(review_titles) > 0 and len(review_bodies) > 0:
        json_results = {"title": review_titles,
                        "review": review_bodies,
                        # "star_ratings": star_ratings,
                        # "votes": votes,
                        "verified": verifications}
        data = pd.DataFrame.from_dict(json_results)
        data['title'] = data['title'].apply(lambda x: x.replace('\n', ''))
        data = data[data['verified'] == True]
        # only data with more than 1 vote (if its bigger, not that many comments)
        #data = data[data['votes'] >= 1]
        data = preprocessing_input(data)
        save_as_csv(data)

        return data
    else:
        print("No scraped Data... Return Nothing")
        return None
