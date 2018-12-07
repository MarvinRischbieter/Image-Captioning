import os
from PIL import Image
from vocabulary import Vocabulary
from torchvision import transforms

class Predictor:
    def __init__(self, vocab_path, model, device):
        self.vocab = Vocabulary(5, vocab_file=vocab_path,
                                vocab_from_file=True)
        self.encoder, self.decoder = model
        self.transform = transforms.Compose([ 
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        self.device = device
        self.encoder.to(device)
        self.decoder.to(device)

    def __call__(self, img_path):
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        features = self.encoder(image.to(self.device)).unsqueeze(1)
        output = self.decoder.sample(features)
        sentence = self._clean_sentence(output)
        return sentence

    def _clean_sentence(self, output):
        sentence = ''
        for idx in output:
            if idx == 0:
                pass
            elif idx == 1:
                sentence = sentence[:-1]
                break
            else:
                ch = self.vocab.idx2word[idx]
                if ch == '.':
                    sentence = sentence[:-1]
                sentence += ch+' '
        return sentence
