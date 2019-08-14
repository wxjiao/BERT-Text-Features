#########################################################################################
# Script name: mustard_bert.py
# Author: Jiao Wenxiang
# Date: 2019-08-12
# Function:
#       1. Extract BERT features for transcripts
#       HOWTO:
#       a. Reconstruct the utterances from the tokens splitted by ourselves, and save them line by line in a .txt file for each video;
#       b. Reprocess our tokens by BERT tokenizer, and record the range of indexes of our tokens in the reconstruted utterance;
#       4. Extract BERT features by the python script and align the features to our tokens based on the recorded ranges.
#########################################################################################

import os
import tqdm
import pickle
import jsonlines
import simplejson as json
import numpy as np
from mustard_extract_features import BertTokenizer,extract_features


import logging
logging.basicConfig(level=logging.INFO)


text_path = "./mustard_dd.json"
tokenizer = BertTokenizer.from_pretrained('./pretrained_model_bert', do_lower_case=True)


def loadFrJson(path):
	file = open(path, 'r')
	obj = json.load(file)
	file.close()

	return obj


def saveToPickle(path, object):
	file = open(path, 'wb')
	pickle.dump(object, file)
	file.close()

	return 1


# Recosntruct utterance from the tokens
def reconUtter(dict_path, input_dir, output_dir):
	data_dict = loadFrJson(dict_path)
	if not os.path.isdir(input_dir):
		os.makedirs(input_dir)
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)
	for vid,vdata in data_dict.items():
		print("Reconstructing {}".format(vid))
		textname = vid + ".txt"
		input_file = os.path.join(input_dir, textname)
		n_segs = len(vdata['data'])
		with open(input_file, 'w') as f:
			for seg_id in range(n_segs):
				# Collect the tokens in an utterance
				utt = []
				for w in vdata['data'][str(seg_id)]:
					utt += w['word']
				rc_utt = " ".join(utt)
				f.write(rc_utt + "\n")
		bertname = vid + "_bert.jsonl"
		output_file = os.path.join(output_dir, bertname)
		extract_features(input_file=input_file, output_file=output_file, bert_model="./pretrained_model_bert", do_lower_case=True)


# Align BERT tokens to our tokens and initialize BERT features
def alignTokens(dict_path, bert_dir, feat_name="mustard_bert.pt"):
	data_dict = loadFrJson(dict_path)
	for vid,vdata in tqdm.tqdm(data_dict.items(), ncols=100, ascii=True):
		bertname = vid + "_bert.jsonl"
		bert_file = os.path.join(bert_dir, bertname)
		with jsonlines.open(bert_file, 'r') as reader:
			# Visit all utterances
			for seg_id,obj in enumerate(reader):
				# Visit all tokens in each utterance
				# [CLS]: 0
				start_idx = 1
				for w in vdata['data'][str(seg_id)]:
					rc_phrase = " ".join(w['word'])
					bert_tokens = tokenizer.tokenize(rc_phrase)
					end_idx = start_idx + len(bert_tokens)
					features = obj['features'][start_idx:end_idx]
					feat_our_token = []
					feat_real_tokens = []
					# Features of tokens by BertTokenizer
					for feature in features:
						feat_real_tokens.append(feature["token"])           # To check if the indexes of features are right
						# Feature of each bert token
						feat_bert_token = []
						# 4 layers
						for layer in [0,1,2,3]:
							feat_bert_token.append(np.array(feature["layers"][layer]["values"]))
						feat_our_token.append(np.mean(feat_bert_token, axis=0))
					w['bert'] = np.mean(feat_our_token, axis=0)
					start_idx = end_idx
					#print(w['word'], bert_tokens, feat_real_tokens)
	saveToPickle(feat_name, data_dict)


def main():
	dict_path = "./MUStARD/mustard_dd.json"
	input_dir = "./MUStARD/bert_input"
	output_dir = "./MUStARD/bert_output"
	# reconUtter(dict_path=dict_path, input_dir=input_dir, output_dir=output_dir)
	alignTokens(dict_path=dict_path, bert_dir=output_dir, feat_name="./MUStARD/mustard_bert.pt")


if __name__ == '__main__':
	main()
