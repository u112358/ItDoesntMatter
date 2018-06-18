import os
import sys
import langid
import re

# _FILE_PATH_ = '../data/negative/data/avatar'
_FILE_PATH_ = '../data/demo/data/avatar'

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print(model.components_)


# def save_img(img_url, file_name, file_path=_FILE_PATH_):
#     try:
#         if not os.path.exists(file_path):
#             print('file', file_path, 'doesn\'t exist, and already established a file')
#             os.makedirs(file_path)
#         file_suffix = os.path.splitext(img_url)[1]
#         filename = '{}{}{}{}'.format(file_path, os.sep, file_name, file_suffix)
#         request.urlretrieve(img_url, filename=filename)
#     except IOError as e:
#         print('file implement error', e)
#     except Exception as e:
#         print('error:', e)

def english_only(text):
    res = langid.classify(text)
    if (res[0] == 'en'):
        return True
    else:
        return False

def format_anew(file_path):
    file = open(file_path, 'r')
    anewobj = {}
    while True:
        line = file.readline()
        if not line:
            break
        
        arr = line.split(' ')
        anewobj[arr[0]] = {}
        anewobj[arr[0]]['valence_mean'] = arr[2]
        anewobj[arr[0]]['valence_sd'] = arr[3][1:-1]
        anewobj[arr[0]]['arousal_mean'] = arr[4]
        anewobj[arr[0]]['arousal_sd'] = arr[5][1:-1]
        anewobj[arr[0]]['dominance_mean'] = arr[6]
        anewobj[arr[0]]['dominance_sd'] = arr[7][1:-1]
    
    return anewobj

def format_wordlist(file_path):
    file = open(file_path, 'r')
    resarr = []
    while True:
        line = file.readline()
        if not line:
            break
        
        str = line.replace('\n', '')
        resarr.append(str)
    
    return resarr


def liwc_count(timeline_data, datalist, type):
    res_count = 0
    for timeline_tweet in timeline_data:
        pattern = re.compile('|'.join(datalist), re.I)
        res = pattern.findall(timeline_tweet['text'])
        res_count += len(res)
    
    return res_count

def domain_count(timeline_data, datalist, type):
    res_count = 0
    if type == 0:
        for timeline_tweet in timeline_data:
            pattern = re.compile('|'.join(datalist), re.I)
            res = pattern.findall(timeline_tweet['text'])
            res_count += len(res)

    if type == 1:
        for timeline_tweet in timeline_data:
            for data in datalist:
                pattern = re.compile(data, re.I)
                res = pattern.findall(timeline_tweet['text'])
                res_count += (len(res)*datalist[data])

    return round(res_count,1)


def symptoms_count(timeline_tweet, datalist, type):
    res_count = 0
    for data in datalist:
        pattern = re.compile(data, re.I)
        res = pattern.findall(timeline_tweet['text'])
        res_count += (len(res)*datalist[data])

    return round(res_count, 1)


    
