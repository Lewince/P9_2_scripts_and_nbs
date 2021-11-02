import pandas as pd
import numpy as np

class PopularityRecommender:
    
    def __init__(self, clicks_df, items_df=None):
        self.model_name = 'Popularity'
        self.clicks_df = clicks_df
        self.popularity_df = clicks_df.groupby('click_article_id')['click_country'].size().sort_values(ascending=False).reset_index()
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def article_selection(self, user_id, top_n=10):
        already_consumed = self.clicks_df[self.clicks_df.loc[:,'user_id'] == user_id]
        already_consumed = already_consumed.loc[:,'click_article_id']
        recommendations_df = self.popularity_df[~self.popularity_df['click_article_id'].isin(already_consumed)] \
                               .head(top_n)
        return recommendations_df.click_article_id.tolist()
