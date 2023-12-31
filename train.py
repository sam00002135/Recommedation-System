from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv("DEvideos.csv", encoding='latin-1', dtype={'col1': str, 'col2': int})


features = ['title','index','views','likes', 'category_id', 'channel_title']



def merge_features(row): 
    return row['title']+" "+str(row['index'])+" "+str(row['views'])+" "+str(row['likes'])+" "+str(row['category_id'])+" "+str(row['channel_title'])

def combine_features(row):
    return row['title']+" "+str(row['index'])+" "+str(row['views'])+" "+str(row['likes'])+" "+str(row['category_id'])+" "+str(row['channel_title'])


for feature in features:
    df[feature] = df[feature].fillna('')
df["combined_features"] = df.apply(combine_features,axis=1)
cv = CountVectorizer() 
count_matrix = cv.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(count_matrix)

def find_title_from_index(index): 
    return df[df.index == index]["title"].values[0] 
def find_index_from_title(title): 
    return df[df.title == title]["index"].values[ 0]

video = input("please enter a video name：")
video_index = find_index_from_title(video)

similar_videos = list(enumerate(cosine_sim[video_index]))

sorted_similar_videos = sorted(similar_videos, key=lambda x: x[1], reverse=True)[1:]

i=0
for element in sorted_similar_videos:
    print(find_title_from_index(element[0]))
    i=i+1
    if i>5:
        break