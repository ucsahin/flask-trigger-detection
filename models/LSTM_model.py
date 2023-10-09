import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM(nn.Module):
    def __init__(self, input_size=768, batch_size=8, hidden_size=100, out_size=1, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, out_size)
        # (H, C)
        self.hidden = (torch.zeros(num_layers,batch_size,hidden_size), torch.zeros(num_layers,batch_size,hidden_size))

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(seq, self.hidden)
        lin1_out = F.relu(self.linear(lstm_out))
        pred = self.linear2(lin1_out)

        return pred[:,-1,:]


def pred_labels(model, test_embeds, num_layers=1, batch_size=1):
    # validation after each epoch ends
    predictions = np.zeros(len(test_embeds))
    pred_probs = np.zeros(len(test_embeds))

    for v in range(0, len(test_embeds), batch_size):
        seq_padded = pad_sequence([torch.Tensor(arr) for arr in test_embeds[v:v+batch_size]], batch_first=True,
                                        padding_value=0.0).to(device)

        with torch.no_grad():
            model.hidden = (
                torch.zeros(num_layers, batch_size, model.hidden_size).to(device),
                torch.zeros(num_layers, batch_size, model.hidden_size).to(device))

            pred_logits = torch.squeeze(model(seq_padded))

            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(pred_logits).cpu().numpy()
            # next, use threshold to turn them into integer predictions
            test_preds = np.zeros(probs.size)
            test_preds[np.where(probs >= 0.5)] = 1

            predictions[v:v+batch_size] = test_preds
            pred_probs[v:v+batch_size] = probs

    return predictions, pred_probs







