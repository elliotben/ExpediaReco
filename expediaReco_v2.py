# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 14:02:47 2017

@author: Bensabat Elliot
"""


# coding: utf-8
import pandas as pd
import os

os.chdir('C:/Users/Bensabat Elliot/Desktop/Expedia')
#destinations = pd.read_csv("destinations.csv")
train = pd.read_csv("train10000.csv")

# In[ ]:
train["date_time"] = pd.to_datetime(train["date_time"])
train["year"] = train["date_time"].dt.year
train["month"] = train["date_time"].dt.month

train["date"] = train["date_time"].dt.date
train["date"] = pd.to_datetime(train["date"])
sortedtrain = train.sort_values(by='date',ascending=True)

#80% test, 20%train with chronological split
i = int(0.8*len(sortedtrain))
endrow = sortedtrain.index[i]
month = sortedtrain.get_value(endrow,'date').month
year = sortedtrain.get_value(endrow,'date').year
x =sortedtrain.get_value(endrow,'date')

x = pd.to_datetime(x) #splitting point
train1 = train[train['date'] <= x]
test1 = train[train['date'] > x]
test1 = test1[test1['is_booking'] == 1]
del train
del sortedtrain 

#%%
#combine predictions of different models
def f5(seq, idfun=None): 
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result

#%%
import operator
def getResults():
    global full_preds
    
    #weighted sum of bookings and clicks in data groupby 'srch_destination_id'
    coef = 0.25
    match_cols = ['srch_destination_id']
    cluster_cols = match_cols + ['hotel_cluster']
    groups = train1.groupby(cluster_cols)
    top_clusters = {}
    for name, group in groups: #name is tuple (65035:10) and group is df for that tuple
        clicks = len(group.is_booking[group.is_booking == False])
        bookings = len(group.is_booking[group.is_booking == True])
        denom = len(group.user_id.unique().tolist())
        score = bookings/denom + coef * clicks
        
        #dictionary with 'srch_destination_id' as key
        #dictionary of 'hotel_cluster' and their score as the value of each key
        clus_name = str(name[0]) 
        if clus_name not in top_clusters:
            top_clusters[clus_name] = {}
        top_clusters[clus_name][name[1]] = score #{'65035':{10:score}}
    
    #weighted sum of bookings and clicks in data groupby 'hotel_market'
    match_cols1 = ['hotel_market'] 
    cluster_cols1 = match_cols1 + ['hotel_cluster']
    groups1 = train1.groupby(cluster_cols1)
    top_clusters1 = {}
    for name, group in groups1: #name is tuple (65035:10) and group is df for that tuple
        clicks1 = len(group.is_booking[group.is_booking == False])
        bookings1 = len(group.is_booking[group.is_booking == True])
        denom1 = len(group.user_id.unique().tolist())
        score1 = bookings1/denom1 + coef * clicks1
        
        #dictionary with 'hotel_market' as key
        #dictionary of 'hotel_cluster' and their score as the value of each key
        clus_name1 = str(name[0]) 
        if clus_name1 not in top_clusters1:
            top_clusters1[clus_name1] = {}
        top_clusters1[clus_name1][name[1]] = score1 #{'65035':{10:score}}        
    
    #get top 5 'hotel_cluster' for each 'srch_destination_id'
    cluster_dict = {}
    for n in top_clusters:
        tc = top_clusters[n]
        top = [l[0] for l in sorted(tc.items(), key=operator.itemgetter(1), reverse=True)[:5]]
        cluster_dict[n] = top
    
    #get top 5 'hotel_cluster' for each 'hotel_market'
    cluster_dict1 = {}
    for m in top_clusters1:
        tc1 = top_clusters1[m]
        top1 = [l[0] for l in sorted(tc1.items(), key=operator.itemgetter(1), reverse=True)[:5]]
        cluster_dict1[m] = top1    
    
    #appends results to test data
    preds = []
    for index, row in test1.iterrows():
        key = str(row[match_cols].item())
        if key in cluster_dict:
            preds.append(cluster_dict[key])
        else:
            preds.append([])
      
    preds1 = []
    for index, row in test1.iterrows():
        key = str(row[match_cols1].item())
        if key in cluster_dict1:
            preds1.append(cluster_dict1[key])
        else:
            preds1.append([])
    
    most_common_clusters = train1.hotel_cluster.value_counts().head().index.tolist()
    full_preds = [f5(preds[p] + preds1[p] + most_common_clusters)[:5] for p in range(len(preds))]
    return(full_preds)

getResults()
#%%
import ml_metrics as metrics
#evaluate accuracy using  Mean Average Precision @ 5
print(metrics.mapk([[l] for l in test1["hotel_cluster"]], full_preds, k=5))