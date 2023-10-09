from flask import Flask, render_template, request
import numpy as np
import torch
from tqdm import tqdm
from models.Transformer_embeddings import construct_embeds
from models.LSTM_model import LSTM, pred_labels
from models.util import load_data, return_labels
import os


# # initialization here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_path = os.path.dirname(__name__)
abs_path = os.path.abspath(current_path) 


def _predict_triggers(text:str):
    df, df_test = load_data(text)

    test_embeds = construct_embeds(df_test)

    # load trained LSTM models here and get label predictions
    predictions = np.zeros([len(test_embeds), 32])
    pred_probs = np.zeros([len(test_embeds), 32])

    for m in (pbar := tqdm(range(32))):
        pbar.set_description(f'LSTM model {m} predictions:')
        lstm_m = LSTM(out_size=1).to(device)
        lstm_m.load_state_dict(torch.load(f'{abs_path}/models/classifiers/model_{m}', map_location=device))

        predictions[:, m], pred_probs[:, m] = pred_labels(lstm_m, test_embeds)

    trigger_dict = return_labels(df['work_id'].values, predictions)

    return trigger_dict, pred_probs


app = Flask(__name__)

@app.route('/trigger-detection', methods=["GET", "POST"])
def index():
    trigger_dict = {'work_id': 0, 'labels': []}
    content = ''

    if request.method == "POST":
        content = request.form.get("trigger-text")

        if content:
            trigger_dict, pred_probs = _predict_triggers(content)
            print(trigger_dict['labels'])
            print(pred_probs)

    return render_template("index.html", content=content, trigger_labels=trigger_dict['labels'])


if __name__ == "__main__":
    app.run()