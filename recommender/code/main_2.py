#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd

from config import config
from modules import *
import sys
# print(sys.version);exit();


def get_aspect_rating():
	ratings_df = pd.read_csv(config.aspect_rating_file)
	ratings_df = ratings_df.rename(columns={"room id":"roomid","star rating":"Overall"})
	ratings_df["comments"] = ratings_df["comments"].astype(str)
	main_idx = 0
	for idx in ratings_df.index:
		roomid = ratings_df.ix[idx,"roomid"]
		comments = ratings_df.ix[idx,"comments"]
		if np.isnan(roomid):
			ratings_df.ix[main_idx,"comments"] += " " + comments
		else: main_idx = idx
	ratings_df = ratings_df[~ratings_df["roomid"].isnull()].reset_index(drop=True)
	ratings_df = ratings_df[["roomid","review count","Overall","comments"]+config.aspect_names]
	ratings_df["roomid"] = ratings_df["roomid"].map(lambda x: str(int(x)))
	# ratings_df = ratings_df.fillna( ratings_df.mean() ) 
	ratings_df = ratings_df.dropna(axis=0)
	return ratings_df
	# print(ratings_df)


def gen_reviews_json():
	
	review_df = get_aspect_rating()
	concat_comments = dict() 
	for idx in review_df.index:
		roomid = review_df.ix[idx,"roomid"]
		rating_dict = dict();
		for aspect_name in config.aspect_names+["Overall"]:
			rating_dict[aspect_name] = review_df.ix[idx,aspect_name]
		comments = review_df.ix[idx,"comments"]
		review_count = review_df.ix[idx,"review count"]
		if concat_comments.get(roomid) is None:
			concat_comments[roomid] = {
				"ratings":rating_dict,
				"comments":comments,
				"review count":review_count,
			}
		else:
			concat_comments[roomid]["comments"] += comments
	json_ans = dict();
	json_ans["comments"] = []
	for roomid,info_dict in concat_comments.items():
		review = dict()
		review["review_num"] = info_dict["review count"]
		review["Ratings"] = info_dict["ratings"]
		review["ReviewID"] = roomid
		review["Content"] = info_dict["comments"]
		json_ans["comments"].append(review)
	store_json(config.reviews_json_file,json_ans)

def get_Beta():
	dictionary_df = pd.read_csv(config.dictionary_file,header=None)
	dictionary_df.columns = ["word"]+["p_%d"%(i) for i in range(8)]
	# print(dictionary_df.ix[:10,:])
	beta_dict = dict()
	for idx in dictionary_df.index:
		word = dictionary_df.ix[idx,"word"]
		beta = 0.0
		for i in range(8):
			beta += dictionary_df.ix[idx,"p_%d"%(i)]
		beta_dict[word] = beta 
	# print(beta_dict)
	return beta_dict


def step1():
	gen_reviews_json()
	print("End csv to json.")

def step2():
	
	rd = ReadData()
	rd.readAspectSeedWords()
	rd.readStopWords()
	rd.readReviewsFromJson()
	rd.removeLessFreqWords()
	print("End clean data.")

	
	bootstrapObj = BootStrap(rd)
	bootstrapObj.bootStrap()
	bootstrapObj.populateLists()
	store_json(config.lrr_in_root_dir+"wList.json",bootstrapObj.wList)
	store_json(config.lrr_in_root_dir+"ratingsList.json",bootstrapObj.ratingsList)
	store_json(config.lrr_in_root_dir+"reviewIdList.json",bootstrapObj.reviewIdList)
	store_json(config.lrr_in_root_dir+"vocab.json",list(bootstrapObj.corpus.wordFreq.keys()))
	store_json(config.lrr_in_root_dir+"aspectKeywords.json",bootstrapObj.corpus.aspectKeywords)
	print("End bootstrap.")

def step3():
	print("LRR model:"),
	beta_ = get_Beta() 
	np.seterr(all='raise')
	model = LRR(beta_)
	model.em_algorithm(maxIter=20, coverge=1e-5) 
	model.output_alpha_and_S(config.lrr_out_root_dir+"weight_ratings.json") 
	model.output_top_k(config.lrr_out_root_dir+"term_weight_topk.json",config.top_k) 
	model.testing_all_data()

def step4():
	aspect_df = get_aspect_rating()
	table6 = Table6_Modles()
	table6.local_model(aspect_df)
	table6.global_model(aspect_df)
	table6.SVR_O(aspect_df)
	table6.SVR_A(aspect_df)

def main():
	step1()
	step2()
	step3()
	step4()

if __name__ == '__main__':
	main()



















