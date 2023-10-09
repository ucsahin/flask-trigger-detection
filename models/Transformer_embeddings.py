import requests
import json
import numpy as np
from dotenv import load_dotenv
import os

current_path = os.path.dirname(__name__)
abs_path = os.path.abspath(current_path) 

load_dotenv(os.path.join(abs_path, '.env'))

API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/ucsahin/roberta-base-trigger"
headers = {"Authorization": "Bearer " + os.environ.get("hf_key")}

def query(payload):
	data = json.dumps(payload)
	response = requests.request("POST", API_URL, headers=headers, data=data)
	return json.loads(response.content.decode("utf-8"))


def construct_embeds(df):
    max_text_id = df['text_id'].values.max() + 1
    seq_embed_list = [None] * max_text_id

    for i, group in (df.groupby('text_id')):
        texts = group['text'].values.tolist()

        output = query({
            "inputs": texts,
            "options": {'wait_for_model': True},
        })

        seq_embed_list[i] = np.array([(output[t][0][0]) for t in range(len(output))])

    return seq_embed_list



