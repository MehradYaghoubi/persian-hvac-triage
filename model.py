"""Small character CNN for reported symptoms, never root-cause diagnosis."""
import json
from pathlib import Path
import re
import torch
from torch import nn

LABELS = ['hot_water', 'heating', 'leak', 'error_code', 'noise', 'maintenance', 'installation', 'price']
ALPHABET = ' ابتثجحخدذرزژسشصضطظعغفقکگلمنوهیءآؤئۀة0123456789abcdefghijklmnopqrstuvwxyz؟!.'
VOCAB = {c:i+1 for i,c in enumerate(ALPHABET)}
def normalize(s):
    return re.sub(r'\s+', ' ', s.translate(str.maketrans('يك۰۱۲۳۴۵۶۷۸۹', 'یک0123456789')).replace('\u200c',' ').lower()).strip()
def encode(text): return [VOCAB.get(c,0) for c in normalize(text)[:160]] + [0]*max(0,160-len(normalize(text)))
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(len(VOCAB)+1,24,padding_idx=0)
        self.conv=nn.Conv1d(24,48,3,padding=1)
        self.head=nn.Linear(48,len(LABELS))
    def forward(self,x): return self.head(torch.relu(self.conv(self.embedding(x).transpose(1,2))).amax(dim=2))
class Predictor:
    def __init__(self, directory):
        folder=Path(directory)
        self.meta=json.loads((folder/'metadata.json').read_text())
        self.model=TinyCNN()
        self.model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True))
        self.model.eval()
    def predict(self,text):
        with torch.no_grad(): probs=self.model(torch.tensor([encode(text)])).softmax(-1)[0]
        score,index=probs.max(0)
        return {'label':LABELS[index] if float(score)>=self.meta['threshold'] else 'review', 'score':round(float(score),4), 'advisory_only':True}
