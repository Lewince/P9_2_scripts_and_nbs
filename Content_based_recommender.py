import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


class CBF_Recommender():

    def __init__(self, initial_clicks_db, user_indexed_clicks_db, article_features):
        self.model_name = 'Content-based filtering'
        self.initial_clicks_db = initial_clicks_db
        self.clicks_db = user_indexed_clicks_db
        self.article_features = article_features
        self.dup_list = self.initial_clicks_db.duplicated(subset=['user_id','click_article_id']) 
    
    def cb_article_weights(self):
        clicks_db = self.initial_clicks_db.copy()
        clicks_db.loc[:,'cbr_weight'] = 1
        clicks_db.cbr_weight[self.dup_list] = clicks_db[self.dup_list].loc[:,'cbr_weight'].apply(lambda x : x + 2)
        return clicks_db.loc[:,'cbr_weight']

    def build_users_profile(self, user_id):
        clicks_weights = self.initial_clicks_db.copy()
        clicks_weights.loc[:,'weights'] = self.cb_article_weights()
        user_clicks_df = clicks_weights[clicks_weights.loc[:,'user_id'] == user_id]
        user_item_profiles = self.article_features.loc[user_clicks_df.drop_duplicates(subset='click_article_id')['click_article_id'],:]
        # print(f'user item profiles shape : {user_item_profiles.shape}')
        user_weights = np.array(user_clicks_df.groupby('click_article_id').agg('sum').loc[:,'weights']).reshape(-1,1)
        # print(f'user weights shape : {user_weights.shape}')
        weighted_user_item_profiles = user_item_profiles.multiply(user_weights)
        user_profile_norm = pd.DataFrame(normalize(weighted_user_item_profiles), index = [user_item_profiles.index], columns=user_item_profiles.columns)
        return user_profile_norm

    def build_users_profiles(self): 
        clicks_db = self.clicks_db.copy()
        interactions_indexed_df = clicks_db[clicks_db['click_article_id'] \
                                                       .isin(self.article_features.index)]
        user_profiles = {}
        for user_id in clicks_db.index.unique():
            user_profiles[user_id] = self.build_users_profile(user_id)
        return user_profiles

    def article_selection(self, user_id, top_n=50, verbose=False):
        user_profile = self.build_users_profile(user_id)
        cosine_similarities = cosine_similarity(user_profile, self.article_features)
        selected_indices = cosine_similarities.argsort().flatten()
        selection = sorted([(i, cosine_similarities[0,i]) for i in selected_indices if i not in user_profile.index.tolist()], key=lambda x: -x[1])        
        if verbose : 
            display(pd.DataFrame(selection))
        return [i[0] for i in selection[top_n:]]

    def display_selection_meta(self, user_selection, meta):
        return meta[meta.loc[:,'article_id'].isin(user_selection)]