import importlib

import torch
import torch.nn as nn

# Import model from torchvision
def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    return getattr(m, class_name)

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, encoder='resnet50'):
        super(EncoderCNN, self).__init__()
        resnet = class_for_name("torchvision.models", encoder)(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size,
                 vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # Turns words into a vector of a specified size
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # Exclude the <end> word to do not predicting when <end> is the input of the RNN
        captions = captions[:, :-1]
        embedding = self.embeddings(captions)
        inputs = torch.cat((features.unsqueeze(1), embedding), dim=1)
        #self.hidden = self.init_hidden(features.size(0))
        lstm_output, _ = self.lstm(inputs, None)
        output = self.linear(lstm_output)
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        tokens = []
        lstm_state = None
        for _ in range(max_len):
            lstm_output, lstm_state = self.lstm(inputs, lstm_state)
            out = self.linear(lstm_output)
            pred = torch.argmax(out, dim=2)
            pred_idx = pred.item()
            tokens.append(pred_idx)

            if pred_idx == 1:  # Stop if <end>
                break
            inputs = self.embeddings(pred)
        return tokens
