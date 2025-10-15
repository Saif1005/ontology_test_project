
"""
medical_ontology_robust.py
Version robuste : entraînement hiérarchique avec cross-entropy + contraintes ontologiques.
"""


import math 
import torch ,torch.nn as nn , torch.nn.functional as F
import random 
from collections import defaultdict, deque

# ============================================================
# 1️⃣  Ontologie médicale
# ============================================================
ONTOLOGY = {
    "Body": {
        "RespiratorySystem": {"Lung": {"Disease": ["Pneumonia","Nodule","Embolism","Fibrosis"]}},
        "CardiovascularSystem": {
            "Heart": {
                "Valve": {
                    "MitralValve": ["Regurgitation","Stenosis"],
                    "AorticValve": ["Stenosis","Insufficiency"]
                },
                "Disease": ["Arrhythmia","MyocardialInfarction"]
            }
        },
        "DigestiveSystem": {
            "Stomach": {"Disease": ["Ulcer","Gastritis","Cancer"]},
            "Liver": {"Disease": ["Cirrhosis","Hepatitis","Steatosis"]}
        },
        "NervousSystem": {"Brain": {"Disease": ["Stroke","Tumor","Epilepsy"]}}
    }
}

def enumerate_leaf_concepts(tree,prefix=None):
    prefix=prefix or []; out=[]
    for k,v in tree.items():
        if isinstance(v,dict):
            out+=enumerate_leaf_concepts(v,prefix+[k])
        elif isinstance(v,list):
            for leaf in v: out.append(".".join(prefix+[k,leaf]))
    return out

LEAF_LIST=enumerate_leaf_concepts(ONTOLOGY)
LEAF2ID={c:i for i,c in enumerate(LEAF_LIST)}
NUM_CONCEPTS=len(LEAF_LIST)

# ============================================================
# 2️⃣  Graphe et matrice de distances
# ============================================================
def edges(tree,prefix=None):
    prefix=prefix or[]; e=[]
    for k,v in tree.items():
        n=".".join(prefix+[k])
        if isinstance(v,dict):
            for c in v.keys(): e.append((n,".".join(prefix+[k,c])))
            e+=edges(v,prefix+[k])
        elif isinstance(v,list):
            for leaf in v: e.append((n,".".join(prefix+[k,leaf])))
    return e

def all_nodes(tree,prefix=None,acc=None):
    prefix=prefix or[]; acc=acc or set()
    for k,v in tree.items():
        n=".".join(prefix+[k]); acc.add(n)
        if isinstance(v,dict): all_nodes(v,prefix+[k],acc)
        elif isinstance(v,list):
            for leaf in v: acc.add(".".join(prefix+[k,leaf]))
    return sorted(acc)

EDGES=edges(ONTOLOGY)
NODES=all_nodes(ONTOLOGY)
adj=defaultdict(list)
for u,v in EDGES: adj[u].append(v); adj[v].append(u)

def sp(u,v):
    if u==v: return 0
    q=deque([u]); dist={u:0}
    while q:
        x=q.popleft()
        for y in adj[x]:
            if y not in dist:
                dist[y]=dist[x]+1
                if y==v: return dist[y]
                q.append(y)
    return math.inf

D=torch.zeros(NUM_CONCEPTS,NUM_CONCEPTS)
for i,a in enumerate(LEAF_LIST):
    for j,b in enumerate(LEAF_LIST): D[i,j]=sp(a,b)
D/=D.max()

# ============================================================
# 3️⃣  Dataset
# ============================================================
DATASET=[
("CT shows nodule in right upper lobe","Body.RespiratorySystem.Lung.Disease.Nodule"),
("Patient has pneumonia with fever","Body.RespiratorySystem.Lung.Disease.Pneumonia"),
("CT indicates pulmonary embolism","Body.RespiratorySystem.Lung.Disease.Embolism"),
("Fibrotic pattern in both lungs","Body.RespiratorySystem.Lung.Disease.Fibrosis"),
("Echocardiogram shows mitral valve regurgitation","Body.CardiovascularSystem.Heart.Valve.MitralValve.Regurgitation"),
("Aortic valve stenosis confirmed","Body.CardiovascularSystem.Heart.Valve.AorticValve.Stenosis"),
("Arrhythmia detected","Body.CardiovascularSystem.Heart.Disease.Arrhythmia"),
("Myocardial infarction on ECG","Body.CardiovascularSystem.Heart.Disease.MyocardialInfarction"),
("Endoscopy reveals stomach ulcer","Body.DigestiveSystem.Stomach.Disease.Ulcer"),
("Signs of gastritis","Body.DigestiveSystem.Stomach.Disease.Gastritis"),
("Liver cirrhosis observed","Body.DigestiveSystem.Liver.Disease.Cirrhosis"),
("Blood test shows hepatitis","Body.DigestiveSystem.Liver.Disease.Hepatitis"),
("Ultrasound shows hepatic steatosis","Body.DigestiveSystem.Liver.Disease.Steatosis"),
("CT scan shows brain tumor","Body.NervousSystem.Brain.Disease.Tumor"),
("MRI confirms stroke","Body.NervousSystem.Brain.Disease.Stroke"),
("EEG indicates epilepsy","Body.NervousSystem.Brain.Disease.Epilepsy"),
("Severe aortic valve insufficiency","Body.CardiovascularSystem.Heart.Valve.AorticValve.Insufficiency"),
("Stomach cancer detected","Body.DigestiveSystem.Stomach.Disease.Cancer"),
("Pulmonary embolism after surgery","Body.RespiratorySystem.Lung.Disease.Embolism"),
("Mitral valve stenosis severe","Body.CardiovascularSystem.Heart.Valve.MitralValve.Stenosis")

]

VOCAB={"<pad>":0,"<bos>":1,"<eos>":2}
for t,_ in DATASET:
    for tok in t.lower().split():
        if tok not in VOCAB: VOCAB[tok]=len(VOCAB)
IVOCAB={i:w for w,i in VOCAB.items()}

def encode(txt,maxlen=25):
    ids=[VOCAB["<bos>"]]+[VOCAB.get(w,0) for w in txt.lower().split()]+[VOCAB["<eos>"]]
    ids=ids[:maxlen]+[VOCAB["<pad>"]]*(maxlen-len(ids))
    return torch.tensor(ids)

# ============================================================
# 4️⃣  Modèle
# ============================================================
class OntoNet(nn.Module):
    def __init__(self,vocab,concepts,d=128):
        super().__init__()
        self.emb=nn.Embedding(vocab,d)
        self.rnn=nn.GRU(d,d,batch_first=True,dropout=0.2)
        self.lm=nn.Linear(d,vocab)
        self.cls=nn.Linear(d,concepts)
        self.c_emb=nn.Embedding(concepts,d)
    def forward(self,x):
        h=self.emb(x)
        out,_=self.rnn(h)
        pooled=out.mean(1)          # mean pooling
        return self.lm(out), self.cls(pooled), pooled

def lm_loss(logits,tgt,pad=0):
    p=logits[:,:-1,:].contiguous().view(-1,logits.size(-1))
    t=tgt[:,1:].contiguous().view(-1)
    return F.cross_entropy(p,t,ignore_index=pad)

def ce_loss(logits,y): return F.cross_entropy(logits,y)
def dist_loss(logits,y,D):
    probs=F.softmax(logits,-1)
    loss=torch.stack([(probs[b]*D[:,y[b]]).sum() for b in range(len(y))])
    return loss.mean()
def align_loss(pooled,emb,y):
    vec=emb(y); cos=F.cosine_similarity(pooled,vec)
    return (1-cos).mean()

# ============================================================
# 5️⃣  Entraînement
# ============================================================
def train():
    dev="cuda" if torch.cuda.is_available() else "cpu"
    model=OntoNet(len(VOCAB),NUM_CONCEPTS).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-2)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,20)
    Dm=D.to(dev)
    batch=8; epochs=40
    alpha,beta,gamma,delta=0.2,1.0,0.5,0.5

    for ep in range(epochs):
        random.shuffle(DATASET)
        for i in range(0,len(DATASET),batch):
            sub=DATASET[i:i+batch]
            X=torch.stack([encode(x) for x,_ in sub]).to(dev)
            Y=torch.tensor([LEAF2ID[y] for _,y in sub]).to(dev)

            lm,cls,vec=model(X)
            L1=lm_loss(lm,X)
            L2=ce_loss(cls,Y)
            L3=dist_loss(cls,Y,Dm)
            L4=align_loss(vec,model.c_emb,Y)
            L=alpha*L1+beta*L2+gamma*L3+delta*L4

            opt.zero_grad(); L.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        scheduler.step()

        with torch.no_grad():
            pred=model(X)[1].argmax(-1)
            acc=(pred==Y).float().mean().item()
        print(f"[epoch {ep}] L_lm={L1:.3f} L_ce={L2:.3f} L_dist={L3:.3f} L_align={L4:.3f} acc={acc:.2f}")

    # -----------------------------------------------------------
    # Inférence
    print("\n[Inference]")
    test="CT shows a lesion in right upper lobe of lung"
    x=encode(test).unsqueeze(0).to(dev)
    with torch.no_grad():
        _,cls,_=model(x)
        p=F.softmax(cls,-1)
        top=p.topk(5,1)
    for k in range(5):
        print(f"{k+1}. {LEAF_LIST[top.indices[0,k]]} (p={top.values[0,k]:.3f})")

if __name__=="__main__":
    train()
