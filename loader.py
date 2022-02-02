import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    data_dir = 'YOURDATADIRECTORYHERE'
    meta = pd.read_csv(data_dir+'/articles_metadata.csv')

    # Ouverture des embeddings, ajout de la catégorie encodée one-hot, du ts de création et de la longueur de l'article normalisés : 
    with open(data_dir+"/articles_embeddings.pickle", "rb") as input_file: 
        embed = pickle.load(input_file)
    scaler = StandardScaler()
    article_features = pd.concat((meta, pd.DataFrame(embed, columns=[f'latent_feature_{n}' for n in range(embed.shape[1])])),
                                 axis=1).set_index('article_id')
    article_features = pd.concat([article_features,
                                  pd.get_dummies(article_features.loc[:,'category_id']).rename(
                                      columns = { n : f'category_{n}' for n in range(len(np.unique(article_features.category_id)))})],
                                 axis=1)
    article_features.loc[:,['created_at_ts','words_count']] = scaler.fit_transform(article_features.loc[:,['created_at_ts','words_count']])
    article_features = article_features.drop(columns=['category_id', 'publisher_id'])

    # Ouverture des données de consommation et création des tables nécessaires 
    clicks_folder = data_dir + '/clicks/clicks'
    clicks_db = pd.DataFrame()
    for csv_file in os.listdir(clicks_folder):
        clicks_db = pd.concat([clicks_db,pd.read_csv(clicks_folder+'/'+csv_file)], ignore_index=True)
    initial_clicks_db = clicks_db.copy()
    clicks_db = clicks_db.set_index('user_id').sort_index(ascending=True)
    
    dup_list = initial_clicks_db.duplicated(subset=['user_id','click_article_id'])
