#-*- coding:utf-8 -*-
import os
import numpy as np
from easydict import EasyDict as edict

if not os.path.exists("./cache/LRR_inputs_jsons/"): os.makedirs("./cache/LRR_inputs_jsons/")
if not os.path.exists("./cache/LRR_outputs_jsons/"): os.makedirs("./cache/LRR_outputs_jsons/")
if not os.path.exists("./logs/"): os.makedirs("./logs/")

config = edict()
# aspect_rating_file 文件中的 aspect name
config.aspect_names = ["Accuracy","Communication","Cleanliness","Location","Check In","Value"]
config.aspect_rating_file = "./data/DSM_review_with_apsect_rating.csv"
config.stopwords_file = "./data/stopwords.dat"
config.seedwords_file = "./data/seedwords_v2.json"
config.dictionary_file = "./data/Dictionary.csv"

config.reviews_json_file = "./cache/review_json_file.json" 
config.lrr_in_root_dir = "./cache/LRR_inputs_jsons/" 
config.lrr_out_root_dir = "./cache/LRR_outputs_jsons/" 
config.lrr_log_file = "./logs/lrr.log"  

config.eps = 1e-8
config.train_test_split_ratio = 1.0 
config.Overall_max_rating = 4.5  
config.aspect_max_rating = 10.0 
config.max_word_polarities = 7.0 
config.top_k = 5 




















