"""Run: python train.py. Small synthetic experiment, not a field benchmark."""
import json
from pathlib import Path
import random
import time
import torch
from model import TinyCNN, LABELS, encode, normalize, Predictor
from data import split_data

def macro_f1(actual,predicted):
    values=[]
    for label in LABELS:
        tp=sum(a==p==label for a,p in zip(actual,predicted)); fp=sum(p==label and a!=label for a,p in zip(actual,predicted)); fn=sum(a==label and p!=label for a,p in zip(actual,predicted))
        values.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return sum(values)/len(values)

def main():
    torch.manual_seed(42);random.seed(42);torch.set_num_threads(1)
    train,val,test=split_data()
    model=TinyCNN(); optimizer=torch.optim.Adam(model.parameters(),lr=.008)
    x=torch.tensor([encode(r['text']) for r in train]); y=torch.tensor([LABELS.index(r['label']) for r in train])
    for _ in range(100):
        optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(model(x),y);loss.backward();optimizer.step()
    model.eval()
    with torch.no_grad():
        vp=model(torch.tensor([encode(r['text']) for r in val])).softmax(-1)
    candidates=[]
    for t in [.3,.4,.5,.6,.7,.8,.9]:
        accepted=[i for i,p in enumerate(vp) if float(p.max())>=t]
        correct=sum(LABELS[int(vp[i].argmax())]==val[i]['label'] for i in accepted)
        if accepted and correct/len(accepted)>=.8:candidates.append((len(accepted),t))
    threshold=max(candidates)[1] if candidates else 1.0
    folder=Path('artifacts');folder.mkdir(exist_ok=True)
    torch.save(model.state_dict(),folder/'model.pt')
    (folder/'metadata.json').write_text(json.dumps({'threshold':threshold,'seed':42,'data':'authored synthetic only'}))
    predictor=Predictor(folder); predicted=[predictor.predict(r['text'])['label'] for r in test]
    # Standard-library baseline: token overlap with training examples.
    def baseline(text):
        q=set(normalize(text).split())
        return max(train,key=lambda r:len(q & set(normalize(r['text']).split())))['label']
    start=time.perf_counter()
    for _ in range(100):predictor.predict('رادیاتور سرد است')
    result={'synthetic_only':True,'train':len(train),'validation':len(val),'test':len(test),'parameters':sum(p.numel() for p in model.parameters()),'weights_bytes':(folder/'model.pt').stat().st_size,'macro_f1':macro_f1([r['label'] for r in test],predicted),'baseline_macro_f1':macro_f1([r['label'] for r in test],[baseline(r['text']) for r in test]),'cpu_mean_ms':(time.perf_counter()-start)*10,'threshold':threshold,'errors':[dict(r,predicted=p) for r,p in zip(test,predicted) if r['label']!=p]}
    (folder/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='errors'}))
if __name__=='__main__':main()
