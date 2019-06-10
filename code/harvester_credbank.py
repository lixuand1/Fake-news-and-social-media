import csv
import json
from multiprocessing import Process
from time import sleep

from tweepy import RateLimitError
from twython import Twython
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

SOURCE_FILE_NAME = ['politifact_real.csv']
# SOURCE_FILE_NAME = ['gossipcop_fake.csv', 'gossipcop_real.csv', 'politifact_fake.csv', 'politifact_real.csv']
RESULT_FILE_NAME = ['text_politifact_real.txt']
# RESULT_FILE_NAME = ['text_gossipcop_fake.txt', 'text_gossipcop_real.txt', 'text_politifact_fake.txt', 'text_politifact_real.txt']
TOPIC_FILE = ['draw_league_arsenal','everyone_sydney_gets', 'justin_padres_upton', 'sony_north_korea', 'sony_north_obama',
              'malaysia_thailand_goal', 'united_villa_falcao', 'shot_police_officers', 'eagles_win_redskins', 'officers_nypd_police',
              'wall_john_acy', 'xbox_squad_down', 'lebron_wade_heat', 'ham_chelsea_west', 'chelsea_than_west',
              'cuomo_mario_york', 'cowboys_game_win', 'ref_flag_pass', 'cowboys_game_lions', 'win_palace_spurs', 'junior_malanda_car',
              'ravens_patriots_flacco','cold_muslim_after','paris_march_rally','stoke_arsenal_sanchez','win_arsenal_city',
              'paris_march_leaders', 'rex_ryan_bills', 'police_paris_post', 'defoe_deal_sunderland', 'wins_joe_state',
              'hope_sunday_arsenal','real_madrid_getafe', 'arsenal_city_man', 'yams_asap_rest', 'brady_colts_tom','blount_game_colts']
# TOPIC_FILE = ['torture_dear_cia','cam_newton_city', 'our_torture_report', 'kendrick_dodgers_howie', 'cespedes_porcello_tigers',
#               'kemp_our_matt', 'trade_sox_well', 'goals_ronaldo_cristiano', 'school_high_shooting', 'music_awards_preview',
#               'goal_chelsea_hazard', 'mariota_win_marcus', 'johnny_manziel_first', 'year_hamilton_lewis', 'malone_coach_kings']
# TOPIC_FILE = ['bowl_pro_odell','costa_diego_charged', 'ebola_obama_night', 'ebola_obama_czar', 'jets_breaking_trade',
#               'jets_seahawks_pick', 'aguero_city_penalty', 'arsenal_hull_team', 'city_aguero_spurs', 'southampton_goal_sunderland',
#               'dame_notre_florida', 'game_fsu_notre', 'bjp_maharashtra_congress', 'years_pistorius_oscar', 'oscar_renta_night']
# TOPIC_FILE = ['senate_keystone_bill','orion_launch_watch', 'world_south_africa','patriots_east_afc', 'year_australia_sydney', 'plane_crash_transasia', 'scott_stuart_tonight']
# TOPIC_FILE = ['host_patrick_neil','total_ceo_crash', 'age_trailer_ultron','police_man_canadian', 'news_senzo_meyiwa', 'king_martin_luther', 'paris_attack_news']
# TOPIC_FILE = ['queen_first_tweet','news_senzo_meyiwa', 'giants_series_win','lance_stephenson_win', 'police_obama_ferguson', 'harbaugh_michigan_jim', 'championship_ohio_national']
CSV_SIZE = 100000000
LEN_LIMIT = 100

csv.field_size_limit(CSV_SIZE)


def api_setup(number):
    comsumer_key = CONSUMER_KEY[number]
    comsumer_secret = CONSUMER_SECRET[number]
    auth_token = OAUTH_TOKEN[number]
    auth_token_secret = OAUTH_TOKEN_SECRET[number]
    auth = tweepy.OAuthHandler(comsumer_key, comsumer_secret)
    auth.set_access_token(auth_token, auth_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    return api


def read_source_file(source_file_name):
    id_file = open(source_file_name, 'r', encoding='UTF-8')
    reader = csv.reader(id_file)
    id_list = []
    for row in reader:
        print(row)
        id_list += row[3].split('\t')
    id_file.close()
    print("Total Tweets: ", len(id_list))
    return id_list


def get_tweet_id_credbank(source_file_name, target_file_name):
    source_file = open(source_file_name, 'r', encoding='UTF-8')
    target_file = open(target_file_name, 'w', encoding='utf-8')
    reader = csv.reader(source_file)
    id_list = []
    i = 0
    for row in reader:
        content = str(row).split(',')
        if i > 0:
            for topic in TOPIC_FILE:
                if topic in row[0]:
                    for ele in content:
                        if 'ID=' in ele:
                            ele = ele.replace(" ('ID=", '')
                            ele = ele.replace("'", '')
                            print(ele)
                            target_file.write(ele)
                            target_file.write('\n')
                    print()
        i += 1
    source_file.close()
    target_file.close()
    return id_list


def harvest_tweet_fakenewsnet():
    for i in range(len(SOURCE_FILE_NAME)):
        id_list = read_source_file(SOURCE_FILE_NAME[i])
        result = open(RESULT_FILE_NAME[i], 'w', encoding="UTF-8")
        j = 0
        l = 0
        error_flag = 0
        for k in range(len(CONSUMER_KEY)):
            api = api_setup(k % len(CONSUMER_KEY))
            while j < len(id_list):
                if j % 1000 == 0:
                    print(j, " lines fetched")
                try:
                    if j+LEN_LIMIT < len(id_list):
                        texts = api.statuses_lookup(id_list[j:j+LEN_LIMIT])
                    else:
                        texts = api.statuses_lookup(id_list[j:len(id_list)])
                    j += LEN_LIMIT
                    error_flag = 0
                    for text in texts:
                        result.write(text.text+'\n')
                        l += 1
                except (RateLimitError, Exception) as e:
                    if isinstance(e, RateLimitError):
                        error_flag += 1
                        print("API ", k, ' reaches limit')
                        if error_flag >= len(CONSUMER_KEY):
                            sleep(20)
                        break
                    else:
                        print(e)
                        pass
        result.close()
        print(l, "lines written to file ", RESULT_FILE_NAME[i])


class SearchHarvester:
    def __init__(self, api_index, tweets_per_query, total_tweets, source_file_name, target_file_name):
        self.api_index = api_index
        self.tweets_per_query = tweets_per_query
        self.source_file_name = source_file_name
        self.target_file_name = target_file_name
        self.total_tweets = total_tweets

    def api_setup(self, number):
        comsumer_key = CONSUMER_KEY[number]
        comsumer_secret = CONSUMER_SECRET[number]
        auth_token = OAUTH_TOKEN[number]
        auth_token_secret = OAUTH_TOKEN_SECRET[number]
        auth = tweepy.OAuthHandler(comsumer_key, comsumer_secret)
        auth.set_access_token(auth_token, auth_token_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        return api

    def get_tweet_id(self):
        id_list = []
        with open(self.source_file_name, 'r', encoding='utf-8') as f:
            i = 0
            for line in f:
                tweet_id = line.replace('\n', '')
                tweet_id = tweet_id.replace(' ', '')
                tweet_id = tweet_id.replace('"', '')
                if i in range(self.api_index*self.total_tweets, (self.api_index+1)*self.total_tweets):
                    if i == self.api_index*self.total_tweets:
                        print(i)
                        print(tweet_id)
                    id_list.append(int(tweet_id))
                i += 1
        print(len(id_list))
        return id_list

    def harvester(self):
        id_list = self.get_tweet_id()
        tweet_count = 0
        api = self.api_setup(self.api_index)
        with open(self.target_file_name, 'w', encoding='utf-8') as f:
            for i in range(int(self.total_tweets/self.tweets_per_query)):
                try:
                    if tweet_count+self.tweets_per_query < self.total_tweets:
                        texts = api.statuses_lookup(id_list[tweet_count:tweet_count+self.tweets_per_query],
                                                    tweet_mode='extended')
                    else:
                        texts = api.statuses_lookup(id_list[tweet_count:len(self.total_tweets)], tweet_mode='extended')
                    tweet_count += self.tweets_per_query
                    if texts:
                        for tweet in texts:
                            try:
                                text = tweet._json['full_text'].replace('\n', ' ')
                                f.write(text+'\n')
                            except Exception as e:
                                print(e)
                except tweepy.TweepError as e:
                    print(e)
        print(tweet_count)


def main():
    # get_tweet_id_credbank('cred_event_SearchTweets.data', 'credbank_realnews_id5.txt')
    harvester1 = SearchHarvester(0, 100, 700000, 'credbank_realnews_id5.txt', 'credbank_realnews_text1.txt')
    harvester2 = SearchHarvester(1, 100, 700000, 'credbank_realnews_id5.txt', 'credbank_realnews_text2.txt')
    harvester3 = SearchHarvester(2, 100, 700000, 'credbank_realnews_id5.txt', 'credbank_realnews_text3.txt')
    harvester4 = SearchHarvester(3, 100, 700000, 'credbank_realnews_id5.txt', 'credbank_realnews_text4.txt')

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