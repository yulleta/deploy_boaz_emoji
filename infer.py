import pandas as pd
import numpy as np
import torch
import pickle
import faiss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import copy

# load
with open('./vector_set.pickle', 'rb') as f:
    vector_set = pickle.load(f)

vectors = []
df = pd.read_csv('emoji_vector_unicode.csv')

vector_set = np.array(vector_set)
vector_set = vector_set.astype('float32')

tokenizer = AutoTokenizer.from_pretrained("augustinLib/text-emoji-encoder-MSMARCO")
model = AutoModelForSequenceClassification.from_pretrained("augustinLib/text-emoji-encoder-MSMARCO")

def inference(save_list):
  input_sequence = copy.deepcopy(save_list)
  print(input_sequence)
  
  # emoji_stacks = []

  tokenized_input = tokenizer(input_sequence, return_tensors="pt", padding='max_length')
  logits = model(input_ids = tokenized_input.input_ids,
                attention_mask = tokenized_input.attention_mask).logits

  query_vector = logits.detach().numpy()

  dimension = 300
  faiss_index = faiss.IndexFlatIP(dimension)
  faiss_index = faiss.IndexIDMap(faiss_index)
  # print(faiss_index.ntotal)

  faiss_index.add_with_ids(vector_set, np.arange(len(vector_set)))

  Distance, Index = faiss_index.search(query_vector, 10)
  # emoji_stacks.append(df.loc[Index[0], :])
  # return_df = pd.concat(emoji_stacks, axis=0).reset_index(drop=True)
  
  return_df = df.loc[Index[0], :]
  
  return return_df