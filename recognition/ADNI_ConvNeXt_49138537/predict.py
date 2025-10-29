import os, json, random, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from dataset import ADNI2p5DTrainSlices, ADNI2p5DEvalSubjects
from modules import ConvNeXtAD, make_criterion

def split_subjects(root, json_name, seed=2025, ratio=(0.7,0.1,0.2)):
    with open(os.path.join(root, json_name), 'r') as f:
        meta = json.load(f)
    subs = [k for k,v in meta.items() if int(v['label']) in (0,2)]
    rng = random.Random(seed); rng.shuffle(subs)
    n = len(subs); n_tr = int(n*ratio[0]); n_val = int(n*ratio[1])
    return subs[:n_tr], subs[n_tr:n_tr+n_val], subs[n_tr+n_val:]

def class_weights(train_ids, root, json_name):
    with open(os.path.join(root, json_name), 'r') as f:
        meta = json.load(f)
    cnt={0:0,1:0}
    for sid in train_ids:
        y = 1 if int(meta[sid]['label'])==0 else 0
        cnt[y]+=1
    total = cnt[0]+cnt[1]
    return [total/(2*max(1,cnt[0])), total/(2*max(1,cnt[1]))]

def train_one(model, loader, loss_fn, opt, dev):
    model.train(); L=[]
    for xb,yb,_ in loader:
        xb,yb=xb.to(dev),yb.to(dev)
        out=model(xb); loss=loss_fn(out,yb)
        opt.zero_grad(); loss.backward(); opt.step()
        L.append(loss.item())
    return np.mean(L)

def eval_slice(model, loader, dev):
    model.eval(); ys,ps=[],[]
    with torch.no_grad():
        for xb,yb,_ in loader:
            xb=xb.to(dev)
            prob=torch.softmax(model(xb),dim=1)[:,1]
            ys+=yb.numpy().tolist(); ps+=prob.cpu().numpy().tolist()
    acc=accuracy_score(ys,(np.array(ps)>=0.5).astype(int))
    try: auc=roc_auc_score(ys,ps)
    except: auc=float('nan')
    return acc,auc

def eval_subject(model, ds, dev):
    model.eval(); ys,ps=[],[]
    ld=DataLoader(ds,batch_size=1,shuffle=False,num_workers=2)
    with torch.no_grad():
        for xg,yb,_ in ld:
            xg=xg.squeeze(0).to(dev)
            p=torch.softmax(model(xg),dim=1)[:,1].mean().item()
            ys.append(int(yb.item())); ps.append(p)
    acc=accuracy_score(ys,(np.array(ps)>=0.5).astype(int))
    try: auc=roc_auc_score(ys,ps)
    except: auc=float('nan')
    return acc,auc

def plot_hist(hist,out):
    plt.figure(figsize=(6,4))
    plt.plot(hist['train_loss'],label='train_loss')
    plt.plot(hist['val_acc'],label='val_acc')
    plt.legend(); plt.tight_layout();
    os.makedirs(out,exist_ok=True)
    plt.savefig(os.path.join(out,'curve.png'),dpi=160)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='/home/groups/comp3710/ADNI')
    ap.add_argument('--json',default='meta_data_with_label.json')
    ap.add_argument('--use_key',default='masked')
    ap.add_argument('--epochs',type=int,default=20)
    ap.add_argument('--batch',type=int,default=32)
    ap.add_argument('--out',default='checkpoints')
    args=ap.parse_args()

    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out,exist_ok=True)
    tr,val,ts=split_subjects(args.root,args.json)
    cw=class_weights(tr,args.root,args.json)
    tr_ds=ADNI2p5DTrainSlices(args.root,args.json,tr,args.use_key)
    val_ds=ADNI2p5DTrainSlices(args.root,args.json,val,args.use_key)
    tr_ld=DataLoader(tr_ds,batch_size=args.batch,shuffle=True,num_workers=4)
    val_ld=DataLoader(val_ds,batch_size=args.batch,shuffle=False,num_workers=4)

    model=ConvNeXtAD().to(dev)
    loss_fn=make_criterion('ce',cw).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)

    hist={'train_loss':[],'val_acc':[]}
    best=0; bp=os.path.join(args.out,'best.pt')
    for ep in range(1,args.epochs+1):
        tl=train_one(model,tr_ld,loss_fn,opt,dev)
        va,_=eval_slice(model,val_ld,dev)
        hist['train_loss'].append(tl); hist['val_acc'].append(va)
        print(f'Epoch {ep}: loss={tl:.4f}, val_acc={va:.4f}')
        if va>best:
            best=va; torch.save({'model':model.state_dict()},bp)
    plot_hist(hist,args.out)
    ts_ds=ADNI2p5DEvalSubjects(args.root,args.json,ts,args.use_key)
    ck=torch.load(bp,map_location=dev)
    model.load_state_dict(ck['model'])
    acc,auc=eval_subject(model,ts_ds,dev)
    print(f'Test acc={acc:.4f}, auc={auc:.4f}')

if __name__=='__main__':
    main()
