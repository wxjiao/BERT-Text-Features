# BERT-Text-Features for Tokenized Transcripts from [P2FA](https://web.sas.upenn.edu/phonetics-lab/)


## Motivation
**Problem**:

Extracting text features from BERT can be accomplished by the provided example script [`extract_features.py`](https://github.com/huggingface/pytorch-transformers/blob/v0.6.2/examples/extract_features.py). However, the problem arises if we want to initialize the words or phrases by the tokenizers that are not integrated in BERT.


**Scenario**:

For example, in multimodal learning, it is a trend to align audios and videos to the corresponding transcripts at word-level. P2FA enables the Forced Alignment between audios and transcripts, where the transcripts are tokenized by its own tokenizer. For convinience, we call this kind of tokens as **P2FA-tokens**, which can be easily initialized by Word2vec or GloVe pre-trained word embeddings because there is no tokenizer requirements. BERT is an architecture, which means we should encode the text by its tokenizer to ensure the correct mapping between the tokens and the features.  


**Solution**:

The solution for aligning the P2FA-tokens to BERT-tokens is as below:
- For Feature Extraction: we reconstruct each utterance from P2FA-tokens and save a .txt file for each video.
```
# P2FA-tokens doesnot split abbreivations of words or punctuations
['you'] ['don't'] ['really'] ['believe'] ['in'] ['that'] ['superstition,'] ['do'] ['you?']

# Some pre-processing for splitting abbreviations of words and punctuations
# This is easy to achieve, and also benefits the initialization via GloVe
# This is the format in our provided `dataset_dd.json` file
['you'] ['don', "'", 't'] ['really'] ['believe'] ['in'] ['that'] ['superstition', ','] ['do'] ['you', '?']

# Reconstrut the utterance
"you don ' t really believe in that superstition , do you ?"
```

- For Token Alignment: we find the index range of each P2FA-token in the BERT-tokens.
```
# BERT-tokens
['[CLS]'] ['you'] ['don'] ["'"] ['t'] ['really'] ['believe'] ['in'] ['that'] ['super'] ['##sti'] ['##tion'] [','] ['do'] ['you'] ['?']

# BERT does sub-word splitting
# A P2FA-token --> BERT-tokens 
['superstition,'] --> ['super'] ['##sti'] ['##tion'] [',']
Index: [9:13)
```
-------------------------------


## Implementation
**Dataset**:

A `dataset_dd.json` file from P2FA with a format as below:
```ruby
{
    "1_60": {
        "data": {
            "0": [
                {
                    "word": [
                        "i"
                    ],
                    "word_idx": [
                        14
                    ],
                    "start_time_clip": 0.0125,
                    "end_time_clip": 0.1525
                },
                {
                    "word": [
                        "never"
                    ],
                    "word_idx": [
                        15
                    ],
                    "start_time_clip": 0.1525,
                    "end_time_clip": 0.5025
                },
                ...
}
```
The .json format output can be obtained by a modified P2FA, named [p2fa-vislab](https://github.com/ucbvislab/p2fa-vislab).


**BERT models**:

Though one can use cloud saved BERT models, we here show how to use locally saved ones.
- First, we need to make a directory named `pretrained_model_bert`.
- Download the files of a BERT model and save them into `pretrained_model_bert`.
- Rename the filenames, taking the `bert-base-uncased` model as an example:
  - `bert-base-uncased-config.json` --> `bert_config.json`
  - `bert-base-uncased-pytorch_model.bin` --> `pytorch_model.bin`
  - `bert-base-uncased-vocab.txt` --> `vocab.txt`


**Run scripts**:

To execute the script:
```ruby
python bert_features.py
```

The `bert_features.py` script is the implementation of the proposed solution, containing a `reconUtter()` function for the first step and an `alignTokens()` function for the second.



