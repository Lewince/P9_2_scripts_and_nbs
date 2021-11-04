from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import train_test_split
import pandas as pd
import numpy as np


class CF_Recommender():
    
    def __init__(self, clicks_data):
        self.model_name = 'Collaborative filtering'
        self.reader = Reader()
        self.clicks_data = clicks_data
        self.dup_list = self.clicks_data.duplicated(subset=['user_id','click_article_id'])
        self.cf_clicks_db = None
        self.data = None
        self.trainset, self.testset = [], []
        self.algo = SVD()
        
    def cf_user_weights(self):
        clicks_db=self.clicks_data
        clicks_db.loc[:,'cf_weight'] = 1
#        clicks_db[self.dup_list]['cf_weight'] += 2
        clicks_db.cf_weight[self.dup_list] = clicks_db[self.dup_list].loc[:,'cf_weight'].apply(lambda x : x + 2)
#         clicks_db.cf_weight[clicks_db['click_country'].isin(clicks_db.loc[[user_id], 'click_country'])] = clicks_db.cf_weight[
#             clicks_db['click_country'].isin(clicks_db.loc[[user_id], 'click_country'])].apply(lambda x : x + 1)
#         clicks_db.cf_weight[clicks_db['click_region'].isin(clicks_db.loc[[user_id], 'click_region'])] = clicks_db.cf_weight[
#             clicks_db['click_region'].isin(clicks_db.loc[[user_id], 'click_region'])].apply(lambda x : x + 1)
        return clicks_db
    
    
    
    def fit_evaluate(self):
        self.cf_clicks_db = self.cf_user_weights()
        # merged = self.cf_clicks_db.merge(pd.DataFrame(self.dup_list, columns=['duplicate']), how = 'left', left_index=True, right_index=True)
        self.cf_clicks_db = self.cf_clicks_db.groupby(['user_id', 'click_article_id'])['cf_weight'].sum().reset_index()
        self.data = Dataset.load_from_df(self.cf_clicks_db, self.reader)
        self.trainset = self.data.build_full_trainset
        self.algo.fit(self.trainset)     
        return self.algo

    def predict_for_user(self, user_id):
        predictions={}
        article_ids = []
        already_consumed = self.clicks_data[self.clicks_data['user_id']==user_id].loc[:,'click_article_id']
        known_users = [self.trainset.to_raw_uid(i) for i in self.trainset.all_users()]
        if user_id in known_users:
            for i in self.trainset.all_items():
                if self.trainset.to_raw_iid(i) not in already_consumed.values:
                    article_ids.append(self.trainset.to_raw_iid(i))
                    predictions[self.trainset.to_raw_iid(i)] = self.algo.predict(user_id, self.trainset.to_raw_iid(i), verbose=False)
        else:
            # use popularity filtering
            print('Unknown_ID, can not make prediction')
        return predictions
    
    def article_selection(self, user_id, top_n=20):
        hits = []
        predictions = self.predict_for_user(user_id)
        for i in predictions.keys():
            hits.append((predictions[i].iid, predictions[i].est))   
        hits.sort(key=lambda x: x[1], reverse=True)
        selection = [hits[i][0] for i in range(len(hits))]
        return selection[:top_n]
        
