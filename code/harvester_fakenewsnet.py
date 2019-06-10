from multiprocessing import Process
import csv
import json
from time import sleep
from tweepy import RateLimitError
import tweepy


CONSUMER_KEY = ["53hobu65awl3VFI9QKZL00vzn", 'IzabACqOqQ2XNshVSjA1lRWHp',
                "VuNPHpkkCpfrvY1gxT9YtNwVC", "DYMWGxnSrF8aG5rISt1oBSBSO"]
CONSUMER_SECRET = ["tnikpvGGh6VbK9FlLBvvGRUrHZR1eNoIrlgnlNUh8KvuHPSB5v",
                   "rzVQDBOggdsO5NsKj6rzKYCRIMEI94t9Ka2XlkICNxg3gnt63i",
                   'loVYUFDnv6eAhZJ13W9sME3nVMftlHasp5V54toUC53KvGqIao',
                   "of33s312AnD247lDcCQGHHK6ciAsdVmqqbm58nwiJo9TAp0lj9"]
OAUTH_TOKEN = ["1123546750742958080-7cfdR4B45YNV7lBTQDgRqZuSNGCfD3", "1117988833436397568-vtgxrL2x0lhJvcPi8tKuRLntAKVqGB",
               "1123726073961832448-0ZhawUlwuD7Bob5F98FbqLRlzUDkXC", "1083653718581497857-VSyJpAIMjFZaWpg0eJ0M8G409KPkJJ"]
OAUTH_TOKEN_SECRET = ["D9XudE2tJ7XGtFZpF1bDMxGL0aghL1om8lvBS1UxaG1zz", "Uwioa9O9RNsA5JygK0bHX84UxsOKiMF283OQpeporN334",
                      "6vP6linzNsB3iRlwrArZHCGzkUzoWQuRmjSXtpvMtr5zC", "3SGS9VfU3UvaXw84y0yRULfdIXFDryxIuxpYD83aMMygP"]


class SearchHarvester:
    def __init__(self, api_index, tweets_per_query, total_tweets, search_query, file_name):
        self.api_index = api_index
        self.sinceId = 1135090000000000000 + api_index * 2000000000000
        self.maxId = self.sinceId + (api_index+1) * 2000000000000
        self.tweets_per_query = tweets_per_query
        self.total_tweets = total_tweets
        self.search_query = search_query
        self.file_name = file_name
        print('since id: ', self.sinceId)
        print('max id: ', self.maxId)

    def api_setup(self, number):
        comsumer_key = CONSUMER_KEY[number]
        comsumer_secret = CONSUMER_SECRET[number]
        auth_token = OAUTH_TOKEN[number]
        auth_token_secret = OAUTH_TOKEN_SECRET[number]
        auth = tweepy.OAuthHandler(comsumer_key, comsumer_secret)
        auth.set_access_token(auth_token, auth_token_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        return api

    def harvester(self):
        tweet_count = 0
        api = self.api_setup(self.api_index)
        with open(self.file_name, 'w', encoding='utf-8') as f:
            while tweet_count < self.total_tweets:
                try:
                    if (self.maxId <= 0):
                        if (not self.sinceId):
                            new_tweets = api.search(q=self.search_query, count=self.tweets_per_query,
                                                    lang='en', tweet_mode='extended')
                        else:
                            new_tweets = api.search(q=self.search_query, count=self.tweets_per_query,
                                                    since_id=self.sinceId, lang='en', tweet_mode='extended')
                    else:
                        if (not self.sinceId):
                            new_tweets = api.search(q=self.search_query, count=self.tweets_per_query,
                                                    max_id=str(self.maxId - 1), lang='en', tweet_mode='extended')
                        else:
                            new_tweets = api.search(q=self.search_query, count=self.tweets_per_query,
                                                    max_id=str(self.maxId - 1),
                                                    since_id=self.sinceId, lang='en', tweet_mode='extended')
                    if not new_tweets:
                        print("No more tweets found")
                        break
                    if new_tweets:
                        for tweet in new_tweets:
                            try:
                                text = tweet._json['retweeted_status']['full_text'].replace('\n', ' ')
                                print(text)
                                f.write(text)
                                f.write('\n')
                                tweet_count += 1
                            except Exception as e:
                                print(e)
                        self.sinceId = new_tweets[-1].id
                        print(self.sinceId)
                except tweepy.TweepError as e:
                    print(e)

        print("Downloaded {0} tweets, Saved to {1}".format(tweet_count, self.file_name))


def main():

    harvester1 = SearchHarvester(0,100,2000000,['#fakenews','#FakeNews'],'search_tweets_1.txt')
    harvester2 = SearchHarvester(1,100,2000000,['#fakenews','#FakeNews'],'search_tweets_2.txt')
    harvester3 = SearchHarvester(2,100,2000000,['#fakenews','#FakeNews'],'search_tweets_3.txt')
    harvester4 = SearchHarvester(3,100,2000000,['#fakenews','#FakeNews'],'search_tweets_4.txt')

    p1 = Process(target=harvester1.harvester)
    p2 = Process(target=harvester2.harvester)
    p3 = Process(target=harvester3.harvester)
    p4 = Process(target=harvester4.harvester)

    p1.start()
    p2.start()
    p3.start()
    p4.start()


if __name__ == '__main__':
    main()