# -*- coding: utf-8 -*-
"""
Created on Thu May 14 19:22:16 2020

@author: Radu
"""

import json
import os


dataRoot = "/Users/Radu/-/Study/MCS-DS/CS-598-DMC/yelp"
yelp_biz = f"{dataRoot}/yelp_academic_dataset_business.json"
yelp_rvw = f"{dataRoot}/yelp_academic_dataset_review.json"


class Business:
    #  businessID: String, name: String, tpe: String, categories: Seq[String]
    def __init__(self, businessID, name, tpe, categories):
        self.businessID = businessID
        self.name = name
        self.tpe = tpe
        self.categories = categories


class Review:
    #  reviewID: String, businessID: String, userID: String, tpe: String, stars: Int, date: LocalDate, text: String
    def __init__(self, reviewID, businessID, userID, tpe, stars, date, text):
        self.reviewID = reviewID
        self.businessID = businessID
        self.userID = userID
        self.tpe = tpe
        self.stars = stars
        self.date = date
        self.text = text

    def __init__(self, json_line):
        json_dict = json.loads(json_line)
        self.reviewID = json_dict['review_id']
        self.businessID = json_dict['business_id']
        self.userID = json_dict['user_id']
        self.tpe = json_dict['type']
        self.stars = json_dict['stars']
        self.date = json_dict['date']
        self.text = self.clean(json_dict['text'])

    def clean(self, text):
        s = str.replace(text, '\n', ' ')\
            .replace('\r', ' ')\
            .replace("\\\"", ' ')\
            .replace("\"", ' ')\
            .replace("\[", ' ')\
            .replace("\]", ' ')
        return str.lower(s)

topCuisines = set(["Mexican", "American (Traditional)", "Fast Food", "Pizza", "Sandwiches", "Italian", "American (New)", "Burgers", "Chinese", "Ice Cream & Frozen Yogurt", "Seafood", "Sushi Bars", "Desserts", "Thai", "Steakhouses", "Greek", "Barbeque", "Japanese", "Asian Fusion", "Juice Bars & Smoothies", "Vietnamese", "Indian", "Chicken Wings", "Korean", "British", "Diners", "Vegetarian", "Mediterranean", "Hot Dogs", "French", "Latin American", "Gluten-Free", "Hawaiian", "Middle Eastern", "Pakistani", "Filipino", "Irish", "Soul Food", "Cajun/Creole", "Fish & Chips", "Soup", "Tex-Mex", "Tapas/Small Plates", "Modern European", "Cheesesteaks", "Caribbean", "Tapas Bars", "Peruvian", "Scottish"])

def parse_businesses(file_path):
    file1 = open(file_path, 'r')
    file2 = open("biz_cuis.csv", 'w')
    count = 0
    while True:
        count += 1
        # Get next line from file
        line = file1.readline()
        # if line is empty, end of file is reached
        if not line:
            break
        biz_dict = json.loads(line)
        business_id = biz_dict['business_id']
        name = biz_dict['name']
        categories = set(biz_dict['categories'])
        cuisines = categories.intersection(topCuisines)
        for cuisine in cuisines:
            try:
                file2.write(f"{business_id}\t{name}\t{cuisine}\n")
            except UnicodeEncodeError:
                print(f"UnicodeEncodeError: [{name}] ")
                pass
        # if cuisines:
        #     print("Line {}: {}".format(count, line.strip()[0:128]))
        #     print(biz_dict['business_id'], cuisines)
        # if count%2500 == 0:
        #     print("Line {}: {}".format(count, line.strip()[0:128]))
        #     print(biz_dict['business_id'], cuisines)
    file1.close()
    file2.close()

def parse_reviews(file_path):
    file1 = open(file_path, 'r')
    count = 0
    while True:
        count += 1
        # Get next line from file
        line = file1.readline()
        # if line is empty, end of file is reached
        if not line:
            break
        if count%25000 == 0:
            print("---------- ---------- ---------- ---------- ---------- ----------")
            review_dict = json.loads(line)
            # print("Line {}: {}".format(count, line.strip()[0:128]))
            print(review_dict['text'][0:256])
            review = Review(line)
            print(review.text[0:256])
    file1.close()


# file_path is the path to "biz_cuis.csv"
def biz_for(file_path, cuisine="Indian"):
    file1 = open(file_path, 'r')
    cuis_biz = {}
    count = 0
    while True:
        count += 1
        # Get next line from file
        line = file1.readline()
        # if line is empty, end of file is reached
        if not line:
            break
        words = line.split('\t')
        business_id = words[0]
        name = words[1]
        cuis = str.replace(words[2], "\n", "")
        if cuis == cuisine:
            cuis_biz[business_id] = name
    file1.close()
    return cuis_biz


def best_restaurants_for(dish_name, cuisine="Indian"):
    indian_restaurants = biz_for("biz_cuis.csv", cuisine)
    rest_ratings = {}
    file1 = open(yelp_rvw, 'r')
    count = 0
    while True:
        count += 1
        # Get next line from file
        line = file1.readline()
        # if line is empty, end of file is reached
        if not line:
            break
        review = Review(line)
        if review.businessID in indian_restaurants and dish_name in review.text:
            rest_name = indian_restaurants[review.businessID]
            if rest_name not in rest_ratings:
                rest_ratings[rest_name] = 0
            rating = rest_ratings[rest_name]
            rest_ratings[rest_name] = rating + review.stars
    file1.close()
    return sort_by_val_desc(rest_ratings)


def write_rest_ratings(dish_name, cuisine="Indian"):
    ratings = best_restaurants_for(dish_name, cuisine)
    module_dir = os.path.dirname(__file__)  # get current directory
    file_name = str.replace(dish_name, " ", "_") + "_py.csv"
    file_path = f'{module_dir}/{file_name}'
    with open(file_path, "w") as out:
        out.write("Restaurant\tRating\n")
        for restaurant, rating in ratings.items():
            out.write(f"{restaurant}\t{rating}\n")
    return file_path


def sort_by_val(dct):
    return {k: v for k, v in sorted(dct.items(), key=lambda item: item[1])}


def neg(dct):
    return {k: -v for k, v in dct.items()}


def sort_by_val_desc(dct):
    return neg(sort_by_val(neg(dct)))


def all_dish_names(cuisine="Indian"):
    module_dir = os.path.dirname(__file__)  # get current directory
    # /some/path/fathomless-reaches-72979/reco/static/reco/indian/indian_dish_names.txt
    dish_names_path = "indian_dish_names.txt"
    with open(dish_names_path) as f:
        dish_names_list = list(f)
    no_newline = list(map(lambda s: str.replace(s, '\n', ''), dish_names_list))
    return list(map(lambda s: str.split(s, '\t')[1], no_newline))


# The most comprehensive list of dishes for the given cuisine
def all_dish_freq(cuisine="Indian"):
    dish_names_path = "indian_dish_names.txt"
    with open(dish_names_path) as f:
        dish_names_list = list(f)
    # For some strange reason, the \n is included at the end of the line
    no_newline = list(map(lambda s: str.replace(s, '\n', ''), dish_names_list))
    dish_freq = {}
    for line in no_newline:
        words = str.split(line, '\t')
        dish_freq[words[1]] = words[0]
    # Now we have {dish_name: frequency}
    return dish_freq





def filter_indian_reviews(cuisine="Indian"):
    import random
    output_file_name = "indian_reviews.json"
    indian_restaurants = biz_for("biz_cuis.csv", cuisine)
    file1 = open(yelp_rvw, 'r')
    file2 = open(output_file_name, 'w')
    count = 0
    while True:
        count += 1
        # Get next line from file
        line = file1.readline()
        # if line is empty, end of file is reached
        if not line:
            break
        review = Review(line)
        if review.businessID in indian_restaurants or random.random() < 0.01:
            file2.write(line)
    file1.close()
    file2.close()
    return output_file_name


def count_indian_reviews(cuisine="Indian"):
    output_file_name = "indian_reviews.json"
    indian_restaurants = biz_for("biz_cuis.csv", cuisine)
    file1 = open(output_file_name, 'r')
    count = 0
    indian = 0
    while True:
        count += 1
        # Get next line from file
        line = file1.readline()
        # if line is empty, end of file is reached
        if not line:
            break
        review = Review(line)
        if review.businessID in indian_restaurants:
            indian = indian + 1
    file1.close()
    return indian


def write_dish_popularity(cuisine="Indian"):
    dish_freq = all_dish_freq(cuisine)
    dish_star = {}
    for dish in dish_freq.keys():
        dish_star[dish] = 0
    indian_restaurants = biz_for("biz_cuis.csv", cuisine)
    review_file_name = "indian_reviews.json"
    output_file_name = "dish_popularity_py.txt"
    file1 = open(review_file_name, 'r')
    file2 = open(output_file_name, 'w')
    count = 0
    while True:
        count += 1
        # Get next line from file
        line = file1.readline()
        # if line is empty, end of file is reached
        if not line:
            break
        review = Review(line)
        if review.businessID in indian_restaurants:
            for dish in dish_freq.keys():
                if dish in review.text:
                    stars = dish_star[dish]
                    dish_star[dish] = stars + review.stars
    file1.close()
    file2.write("Frequency\tDish\tRating\n")
    for dish in dish_freq.keys():
        line = f"{dish_freq[dish]}\t{dish}\t{dish_star[dish]}\n"
        file2.write(line)
    file2.close()
    return output_file_name

write_dish_popularity()

# print(all_dish_freq())
# filter_indian_reviews()
# print(count_indian_reviews())



# print(all_dish_names())

# print(write_rest_ratings("butter chicken"))

# x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
# y = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
# print(y)
# {0: 0, 2: 1, 1: 2, 4: 3, 3: 4}

# x = {1: 3, 2: 5, 3: 8, 4: 9, 5: 1, 6: 6, 7: 3, 8: 7, 9: 2, 10: 4}
# y = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
# print(sort_by_val_desc(x))


# print(best_restaurants_for("tandoori chicken"))

# print(len(biz_for("biz_cuis.csv", "Indian")))

# print(yelp_biz)
# parse_businesses(yelp_biz)

# print(yelp_rvw)
# parse_reviews(yelp_rvw)

# for cuisine in topCuisines:
#     print(cuisine)

# https://www.stechies.com/check-list-array-set-tuple-string-dictionary-empty-python/
# https://stackoverflow.com/questions/8424942/creating-a-new-dictionary-in-python
# https://www.w3schools.com/python/python_json.asp
# https://docs.python.org/3/tutorial/classes.html
# https://docs.python.org/3/tutorial/errors.html#handling-exceptions
# https://www.geeksforgeeks.org/read-a-file-line-by-line-in-python/
# https://stackoverflow.com/questions/1602934/check-if-a-given-key-already-exists-in-a-dictionary
# https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
