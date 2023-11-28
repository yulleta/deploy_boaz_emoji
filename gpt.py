import openai
import os
import json
from openai import OpenAI
import streamlit as st

from translate import Translator
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import re
import ast
import copy

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key= st.secrets["API_KEY"],
)

def gpt_answ(text):
    question = '아래의 일기에서 [기분, 날씨, 행동, 사물, 음식, 장소]의 요소들을 각각 추출해서 json 형태로 답변해줘.'
    # text = input()
    text = copy.deepcopy(text)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": question + text,
            }
        ],
        response_format={ "type": "json_object" },
        model="gpt-3.5-turbo-1106",
        seed = 24
    )

    print(chat_completion)

    question2 = 'key-value 형태의 텍스트에서 key는 그대로 두고 value 부분에 해당하는 것들을 영어로 번역해줘'
    text2 = chat_completion.choices[0].message.content

    chat_completion2 = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": question2 + text2,
            }
        ],
        model="gpt-3.5-turbo-1106",
        seed = 24
    )

    print()
    print(chat_completion2)

    data_dict = ast.literal_eval(chat_completion2.choices[0].message.content)
    
    data_dict_list = [data_dict['기분'], data_dict['날씨'], data_dict['행동'], data_dict['사물'], data_dict['음식'],data_dict['장소']]
    기분 = []
    날씨 = []
    행동 = []
    사물 = []
    음식 = []
    장소 = []
    # save_list에 각 리스트를 추가
    save_list = [기분, 날씨, 행동, 사물, 음식, 장소]
    
    print()
    print(data_dict_list)

    s = 0
    # 텍스트를 ',' 구분자로 나누어 리스트로 변환
    for data_dict_ele in data_dict_list:
        if isinstance(data_dict_ele, str):
            values_list = data_dict_ele.split(', ')
        else:
            values_list = data_dict_ele

        # save_list의 각 리스트에 값을 추가
        for i in range(len(values_list)):
            save_list[s].append(word_tokenize(values_list[i]))
        s+=1

    save_list = [item for sublist in save_list for subsublist in sublist for item in subsublist]
    articles_and_prepositions = ["a", "an", "the", \
        "in", "to", "for", "of", "on", "at", "by", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "from", "up", "down", "out", "off", "over", "under",\
        'and', 'but', 'or', 'I', 'He', 'She', 'They']

    # 이미 token화된 save_list에서 관사와 전치사 제거
    filtered_save_list = list(set([word.lower() for word in save_list if word not in articles_and_prepositions]))

    # 결과 출력
    for saved_data in set(filtered_save_list):
        print(saved_data)

    return filtered_save_list


