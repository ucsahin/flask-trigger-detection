from typing import List
import pandas as pd
from numpy.typing import ArrayLike

# split data text into chunks with overlap
def split_to_chunks(id_list, text_list, chunk_size=200, overlap=50):
    new_text_list = []
    new_work_id_list = []
    new_id_list = []

    for t in range(len(text_list)):
        work_id = id_list[t]
        text = text_list[t]

        tokens = text.split()

        chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size - overlap)]
        new_text_list.extend([' '.join(chunks[i]) for i in range(len(chunks))])

        new_work_id_list.extend([work_id for i in range(len(chunks))])
        new_id_list.extend([t for i in range(len(chunks))])

    return new_work_id_list, new_id_list, new_text_list


#STOPWORDS = set(stopwords.words('english'))
def preprocess_df(df):
    df['text_clean'] = df['text'].str.replace(r'<[^<>]*>', '', regex=True)  # remove html tags
    df['text_clean'] = df['text_clean'].str.replace(r'http\S+', '', regex=True)  # remove urls
    df['text_clean'] = df['text_clean'].str.replace('&nbsp;', ' ')
    df['text_clean'] = df['text_clean'].str.replace('\n', ' ')
    df['text_clean'] = df['text_clean'].str.replace('\n\n', ' ')
    df['text_clean'] = df['text_clean'].str.replace('\n \n', ' ')

    df['text_clean'] = df['text_clean'].str.lower()  # all lower case
    return df


def load_data(text:str):
    df = pd.DataFrame(data={"work_id": 0, "text": text}, index=[0])
    df = preprocess_df(df)

    # test data split to chunks

    new_work_id_list, new_id_list, new_text_list = split_to_chunks(df['work_id'].values,
                                                                   df['text_clean'].values)
    df_test = pd.DataFrame(list(zip(new_work_id_list, new_id_list, new_text_list)),
                           columns=['work_id', 'text_id', 'text'])

    return df, df_test

LABELS = ["pornographic-content", "violence", "death", "sexual-assault", "abuse", "blood", "suicide",
          "pregnancy", "child-abuse", "incest", "underage", "homophobia", "self-harm", "dying", "kidnapping",
          "mental-illness", "dissection", "eating-disorders", "abduction", "body-hatred", "childbirth",
          "racism", "sexism", "miscarriages", "transphobia", "abortion", "fat-phobia", "animal-death",
          "ableism", "classism", "misogyny", "animal-cruelty"]  # 32

def return_labels(work_ids: List[str], labels: ArrayLike):
    for wid, label_list in zip(work_ids, labels):
        result = {"work_id": wid, "labels": [LABELS[idx] for idx, cls in enumerate(label_list) if cls == 1]}

    return result