import os, json, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from dataset import ADNI2p5DEvalSubjects
from modules import ConvNeXtAD

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='/home/groups/comp3710/ADNI')
    ap.add_argument('--json',default='meta_data_with_label.json')
    ap.add_argument('--use_key',default='masked')
    ap.add_argument('--ckpt',default='checkpoints/best.pt')
    args=ap.parse_args()

    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(args.root,args.json),'r') as f: meta=json.load(f)
    subs=[k for k,v in meta.items() if int(v['label']) in (0,2)]
    n=len(subs); test=subs[int(0.8*n):]
    ds=ADNI2p5DEvalSubjects(args.root,args.json,test,args.use_key)
    ld=DataLoader(ds,batch_size=1,shuffle=False)
    model=ConvNeXtAD().to(dev)
    ck=torch.load(args.ckpt,map_location=dev)
    model.load_state_dict(ck['model']); model.eval()
    ys,ps=[],[]
    with torch.no_grad():
        for xg,yb,_ in ld:
            xg=xg.squeeze(0).to(dev)
            p=torch.softmax(model(xg),dim=1)[:,1].mean().item()
            ys.append(int(yb.item())); ps.append(p)
    yhat=(np.array(ps)>=0.5).astype(int)
    acc=(yhat==np.array(ys)).mean()
    fpr,tpr,_=roc_curve(ys,ps); roc_auc=auc(fpr,tpr)
    cm=confusion_matrix(ys,yhat)
    print(f'ACC={acc:.4f}, AUC={roc_auc:.4f}\nConfusion:\n{cm}')
    plt.plot(fpr,tpr,label=f'AUC={roc_auc:.3f}'); plt.legend()
    plt.savefig('checkpoints/roc.png',dpi=160)

if __name__=='__main__':
    main()
