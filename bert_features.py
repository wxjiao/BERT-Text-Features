######
#
######

import tqdm
import simplejson as json

import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

text_path = "./mustard_dd.json"


def LoadFrJson(path):
    file = open(path, 'r')
    object = json.load(file)
    return object
    

# Recosntruct utterance from the tokens
def ReconsUtter(path):
    data_dict = LoadFrJson(path)
    for vid,vdata in tqdm(data_dict, col=100):
        with open(vid + ".txt", 'w') as f:
            for seg_id,seg in vdata['data']:
               utt = []
               for w in seg:
                    utt += w['word']
               rc_utt = " ".join(utt)
               f.write(rc_utt + "\n")
                
                
