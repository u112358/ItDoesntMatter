import os
import json
import sys
import re
import csv
import math
sys.path.append('/usr/local/lib/python3.6/site-packages')
import numpy as np

import datalist.emotion_list as emotions
import datalist.antidepressant_list as antidepressants
import datalist.symptoms as symptoms
import _util

# _POSITIVE_PATH_ = '/Users/vchacha/Dropbox/Supervision/Masters/Junyan/Dataset/labeled/positive/data'
_POSITIVE_PATH_ = '../data/positive/data'
_POSITIVE_OUTPUT_PATH_ = './outputs/positive.csv'
_POSITIVE_OUTPUT_TEN_PATH_ = './outputs/positive_ten.csv'
# _NEGATIVE_PATH_ = '/Users/vchacha/Dropbox/Supervision/Masters/Junyan/Dataset/labeled/negative/data'
_NEGATIVE_PATH_ = '../data/negative/data'
_NEGATIVE_OUTPUT_PATH_ = './outputs/negative.csv'
_NEGATIVE_OUTPUT_TEN_PATH_ = './outputs/negative_ten.csv'
_N_TOPIC_ = 25


class Extract(object):

    def __init__(self,path,output_path, percent=1.0):

        self.path = ''
        self.output_path = output_path
        self.percent = percent

        self.timelines_path = {}

        self.vad_dic = _util.format_anew('./datalist/anew_all.txt')
        self.positive_list = _util.format_wordlist('./datalist/positive_words.txt')
        self.negative_list = _util.format_wordlist('./datalist/negative_words.txt')

        self.create_csvfile()

        self.read_files(path)

    def read_files(self, path):
        if os.path.isdir(path):
            self.path = path
            users_directory = os.path.join(path, 'users')
            # tweet_directory = os.path.join(path, 'tweet')
            timeline_directory = os.path.join(path, 'timeline')
        else:
            print('path is not a directory')
            return 

        print('start reading data')
        """read timeline files"""
        for idx, file in enumerate(os.listdir(timeline_directory)):
            print('timeline data:',idx+1, ' of ', len(os.listdir(timeline_directory)))
            file_path = os.path.join(timeline_directory, file)
            if os.path.isdir(file_path):
                print(file_path + ' is not a file')
            else:
                user_name = file.split('created_at')[0]
                if user_name not in self.timelines_path:
                    self.timelines_path[user_name] = []
                self.timelines_path[user_name].append(file_path)

        print('finish reading the users directory')

        """read users files"""
        for idx, file in enumerate(os.listdir(users_directory)):
            print('user data:',idx+1, ' of ', len(os.listdir(users_directory)))
            file_path = os.path.join(users_directory, file)
            if os.path.isdir(file_path):
                print( file_path + ' is not a file')
            else:
                with open(file_path, 'r') as json_file:
                    user_name = file[:-5]
                    user_data = json.load(json_file)
                    if user_name in self.timelines_path:
                        timeline_datas = []
                        for timeline_path in self.timelines_path[user_name]:
                            timeline_data = open(timeline_path, 'r')
                            with timeline_data as f:
                                for line in f:
                                    timeline_datas.append(json.loads(line))

                        if self.percent != 1:
                            newlen = math.ceil(len(timeline_datas)*self.percent)
                            timeline_datas = timeline_datas[0:newlen]

                        if len(timeline_datas) > 0: 
                            print(len(timeline_datas))
                            self.extract_data(user_data, timeline_datas)

        print('finish reading the timeline directory')
                    
        # """read tweet files"""
        # for file in os.listdir(tweet_directory):
        #     file_path = os.path.join(tweet_directory, file)
        #     if os.path.isdir(file_path):
        #         print( file_path + ' is not a file')
        #     else:
        #         tweet_data = json.loads(open(file_path, 'r').read())
        #         self.tweets_data.append(tweet_data)
        # print('finish reading the tweet directory')

    def extract_data(self, user_data, timeline_datas):
        user_id = user_data["id_str"]
        user_list = []
        ''' check language '''
        if _util.english_only(timeline_datas[0]['text']):
            user_list.append(user_id)
            self.social_network(user_id, user_data, timeline_datas, user_list)
            self.other_features(timeline_datas, user_list)
            # self.topic_level(user_id, timeline_datas, user_list)
            self.write_files(user_list)
                
        print('finish writing the user data features')

        
    def social_network(self, user_id, user_data, timeline_data, user_list):

        """analyse user data"""
        # Social interactions 
        user_list.append(user_data['statuses_count'])
        user_list.append(user_data['followers_count'])
        user_list.append(user_data['friends_count'])
        user_list.append(user_data['listed_count'])
        user_list.append(user_data['favourites_count'])
        user_list.append(user_data['status']['favorite_count'])
        user_list.append(user_data['status']['retweet_count'])

        """analyse timeline data"""
        alltweet_num = len(timeline_data)
        posts = {}
        alltext_len = 0
        reply_num = 0
        for idx, timeline_tweet in enumerate(timeline_data):
            day = timeline_tweet['created_at'][:10]
            hour = int(timeline_tweet['created_at'][11:13])
            alltext_len += len(timeline_tweet['text'].split(" "))
            if timeline_tweet['in_reply_to_status_id']:
                reply_num += 1
            if day in posts:
                posts[day][hour] +=1
            else:
                posts[day] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            
        # Number of tweets (timeline)
        days = len(posts)
        posts_number_all = alltweet_num
        posts_number = round(alltweet_num/days)
        text_length = round(alltext_len/alltweet_num)
        reply_ratio = round(reply_num/alltweet_num, 2)
        user_list.append(days)
        user_list.append(posts_number_all)
        user_list.append(posts_number)
        user_list.append(text_length)
        user_list.append(reply_ratio)

        # Posting behaviors
        posts_time = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        posts_all = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for time in posts.values():
            for idx, num in enumerate(time):
                posts_all[idx] += num
        for idx, hour_num in enumerate(posts_all):
            posts_time[idx] = round((hour_num/alltweet_num),3)
        user_list.append(posts_time)
    

    def other_features(self, timeline_data, user_list):
        # LIWC
        positive_counts = 0
        negative_counts = 0
        # emoji
        emotion_counts = 0
        # ANEW words
        valence_sum = 0
        arousal_sum = 0
        dominance_sum = 0
        valence_value = 0
        arousal_value = 0
        dominance_value = 0
        # antidepressant
        antidepressant_counts = 0
        # depression symptoms
        depressed_mood_count = 0
        decreased_interest_count = 0
        weight_change_count = 0
        sleep_change_count = 0
        activity_change_count = 0
        fatigue_count = 0
        guilt_count = 0
        concentration_count = 0
        suicidality_count = 0

        # main loop
        for timeline_tweet in timeline_data:
            # LIWC
            positive_pattern = re.compile('|'.join(self.positive_list), re.I)
            negative_pattern = re.compile('|'.join(self.negative_list), re.I)
            positive_counts += len(positive_pattern.findall(timeline_tweet['text']))
            negative_counts += len(negative_pattern.findall(timeline_tweet['text']))
            # emoji
            emoji_pattern = re.compile('|'.join(emotions._ALL_), re.UNICODE)
            emotion_counts += len(emoji_pattern.findall(timeline_tweet['text']))
            # ANEW words
            wordarr = re.sub('[^a-zA-Z\' ]', 's', timeline_tweet['text']).split(' ')
            # print(wordarr)
            for word in wordarr:
                if word in self.vad_dic:
                    valence_sum += float(self.vad_dic[word]['valence_mean'])
                    arousal_sum += float(self.vad_dic[word]['arousal_mean'])
                    dominance_sum += float(self.vad_dic[word]['dominance_mean'])
            # antidepressant
            antidepressant_pattern = re.compile('|'.join(antidepressants._ALL_), re.I)
            antidepressant_counts += len(antidepressant_pattern.findall(timeline_tweet['text']))
            # depression symptoms
            depressed_mood_count += _util.symptoms_count(timeline_tweet, symptoms._DEPRESSED_MOOD_, 1)
            decreased_interest_count += _util.symptoms_count(timeline_tweet, symptoms._DECREASED_INTEREST_, 1)
            weight_change_count += _util.symptoms_count(timeline_tweet, symptoms._WEIGHT_CHANGE_, 1)
            sleep_change_count += _util.symptoms_count(timeline_tweet, symptoms._SLEEP_CHANGE_,1)
            activity_change_count += _util.symptoms_count(timeline_tweet, symptoms._ACTIVITY_CHANGE_, 1)
            fatigue_count += _util.symptoms_count(timeline_tweet, symptoms._FATIGUE_, 1)
            guilt_count += _util.symptoms_count(timeline_tweet, symptoms._GUILT_,1)
            concentration_count += _util.symptoms_count(timeline_tweet, symptoms._CONCENTRATION_,1)
            suicidality_count += _util.symptoms_count(timeline_tweet, symptoms._SUICIDALITY_, 1)


        valence_value = valence_sum/len(timeline_data)
        arousal_value = arousal_sum/len(timeline_data)
        dominance_value = dominance_sum/len(timeline_data)
        vad = [round(valence_value,2), round(arousal_value,2), round(dominance_value,2)]
        # add features
        user_list.append(positive_counts)
        user_list.append(negative_counts)
        user_list.append(emotion_counts)
        user_list.append(vad)
        user_list.append(antidepressant_counts)
        user_list.append([
            round(depressed_mood_count,1),
            round(decreased_interest_count,1),
            round(weight_change_count,1),
            round(sleep_change_count,1),
            round(activity_change_count,1),
            round(fatigue_count,1),
            round(guilt_count,1),
            round(concentration_count,1),
            round(suicidality_count,1)
        ])

    
    # def topic_level(self, user_id, timeline_data, user_list):
    #     lda = LatentDirichletAllocation(n_components=_N_TOPIC_,learning_method='online')
    #     text_list = []
    #     for timeline_tweet in timeline_data:
    #         # lower case
    #         text = timeline_tweet['text'].lower()
    #         print(nlp(text))
    #         text_list.append(text)
    #
    #     cntVector = CountVectorizer()
    #     cntTf = cntVector.fit_transform(text_list)
    #     topics = lda.fit_transform(cntTf)
    #     user_list.append(topics)

        """ print top 25 topics """
        # feature_names = cntVector.get_feature_names()
        # _util.print_top_words(lda, feature_names, _N_TOPIC_)

    def create_csvfile(self):
        with open(self.output_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow([
                'user_id',
                'statuses_count',
                'followers_count',
                'friends_count',
                'listed_count',
                'favourites_count',
                'favorite_count',
                'retweet_count',
                'days',
                'posts_number_all',
                'posts_number',
                'text_length',
                'reply_ratio',
                'time',
                'positive_words_count',
                'negative_words_count',
                'emoji_count',
                'vad',
                'antidepressant_words_count',
                'depression_symptoms_count'
            ])

    def write_files(self, user_list):
        with open(self.output_path, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(user_list)


# test = Extract('./demo/data', './outputs/outputs_test.csv')
# test = Extract(_POSITIVE_PATH_, _POSITIVE_OUTPUT_PATH_)
# test1 = Extract(_POSITIVE_PATH_, _POSITIVE_OUTPUT_TEN_PATH_, 0.1)
test2 = Extract(_NEGATIVE_PATH_, _NEGATIVE_OUTPUT_PATH_)
test3 = Extract(_NEGATIVE_PATH_, _NEGATIVE_OUTPUT_TEN_PATH_, 0.1)






