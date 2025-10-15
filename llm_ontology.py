# medical_onto_loss.py
# -------------------------------------------------------------
# Démo pédagogique : L_total = L_LM + λ1 * L_dist + λ2 * L_align
#  - L_LM    : cross-entropy next-token
#  - L_dist  : distance ontologique attendue (soft, différentiable)
#  - L_align : 1 - cos(emb_pred, emb_true) sur les concepts
# -------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque

# ---------------------------
# 1) Ontologie médicale (démo)
# ---------------------------
# Hiérarchie simple => on ne classifie que les FEUILLES (concepts de sortie)
ONTOLOGY = {
    "Body": {
        "CardiovascularSystem": {
            "Heart": {
                "Valve": {
                    "MitralValve": ["Regurgitation", "Stenosis"],
                    "AorticValve": ["Stenosis"]
                }
            }
        },
        "RespiratorySystem": {
            "Lung": {
                "Lobe": ["UpperLobe", "LowerLobe"],
                "Disease": ["Pneumonia", "Nodule", "Embolism"]
            }
        },
        "DigestiveSystem": {
            "Stomach": {
                "Disease": ["Ulcer", "Gastritis"]
            }
        }
    }
}

def enumerate_leaf_concepts(tree, prefix=None):
    """Parcourt l'ontologie et retourne les 'chemins' des concepts FEUILLES en dot-notation."""
    prefix = prefix or []
    leaves = []
    for k, v in tree.items():
        if isinstance(v, dict):
            leaves += enumerate_leaf_concepts(v, prefix + [k])
        elif isinstance(v, list):
            # k est une catégorie, v contient les feuilles
            for leaf in v:
                leaves.append(".".join(prefix + [k, leaf]))
        else:
            # cas rare
            leaves.append(".".join(prefix + [k, str(v)]))
    return leaves

LEAF_LIST = enumerate_leaf_concepts(ONTOLOGY)
LEAF2ID = {c: i for i, c in enumerate(LEAF_LIST)}
NUM_CONCEPTS = len(LEAF_LIST)

# Construire graphe non orienté des concepts (en reliant les noeuds adjacents du chemin)
# Puis distance entre FEUILLES par plus court chemin
def ontology_graph_edges(tree, prefix=None):
    """Retourne des arêtes (u,v) sur les noeuds de l'arbre (chaque segment du chemin)."""
    prefix = prefix or []
    edges = []
    for k, v in tree.items():
        node = ".".join(prefix + [k])
        if isinstance(v, dict):
            for k2 in v.keys():
                child = ".".join(prefix + [k, k2])
                edges.append((node, child))
            edges += ontology_graph_edges(v, prefix + [k])
        elif isinstance(v, list):
            for leaf in v:
                child = ".".join(prefix + [k, leaf])
                edges.append((node, child))
    return edges

# Récupère tous les noeuds (internes + feuilles)
def ontology_all_nodes(tree, prefix=None, acc=None):
    prefix = prefix or []
    acc = acc or set()
    for k, v in tree.items():
        node = ".".join(prefix + [k])
        acc.add(node)
        if isinstance(v, dict):
            ontology_all_nodes(v, prefix + [k], acc)
        elif isinstance(v, list):
            for leaf in v:
                acc.add(".".join(prefix + [k, leaf]))
    return sorted(acc)

ALL_NODES = ontology_all_nodes(ONTOLOGY)
NODE2ID = {n: i for i, n in enumerate(ALL_NODES)}
EDGES = ontology_graph_edges(ONTOLOGY)

# BFS distance sur graphe non orienté
adj = defaultdict(list)
for u, v in EDGES:
    adj[u].append(v)
    adj[v].append(u)

def shortest_path_dist(u, v):
    if u == v:
        return 0
    q = deque([u])
    dist = {u: 0}
    while q:
        x = q.popleft()
        for y in adj[x]:
            if y not in dist:
                dist[y] = dist[x] + 1
                if y == v:
                    return dist[y]
                q.append(y)
    return math.inf  # devrait pas arriver si connectée

# Matrice D_ij = distance entre feuilles i et j
with torch.no_grad():
    D = torch.zeros(NUM_CONCEPTS, NUM_CONCEPTS, dtype=torch.float32)
    LEAF_NODES = LEAF_LIST[:]  # dot-notation
    for i, ci in enumerate(LEAF_NODES):
        for j, cj in enumerate(LEAF_NODES):
            # distance = plus court chemin dans graphe entre feuilles (via noeuds internes)
            D[i, j] = shortest_path_dist(ci, cj)
    # Normalisation (optionnelle) pour stabilité
    if D.max() > 0:
        D = D / D.max()

# ---------------------------------
# 2) Mini vocab + tokenizer naîf
# ---------------------------------
# (Tu peux remplacer par un vrai tokenizer BPE/HF)
VOCAB = {
    "<pad>": 0, "<bos>": 1, "<eos>": 2,
    "ct": 3, "scan": 4, "shows": 5, "lesion": 6, "upper": 7, "lobe": 8, "lung": 9,
    "echocardiogram": 10, "severe": 11, "mitral": 12, "valve": 13, "regurgitation": 14,
    "pneumonia": 15, "lower": 16, "embolic": 17, "aortic": 18, "stenosis": 19,
    "stomach": 20, "ulcer": 21, "gastritis": 22, "nodule": 23, "right": 24, "left": 25,
    "suggest": 26, "biopsy": 27, "recommend": 28, "therapy": 29, "anticoagulant": 30
}
IVOCAB = {i: w for w, i in VOCAB.items()}

def encode(text, max_len=24):
    toks = text.lower().split()
    ids = [VOCAB.get(t, 0) for t in toks]
    ids = [VOCAB["<bos>"]] + ids[:max_len-2] + [VOCAB["<eos>"]]
    pad = [VOCAB["<pad>"]] * (max_len - len(ids))
    return torch.tensor(ids + pad, dtype=torch.long)

# ---------------------------------------------------
# 3) Mini-modèle : embeddings + GRU + 2 têtes (LM/Concept)
# ---------------------------------------------------
class MiniOntoLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_concepts=NUM_CONCEPTS):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.lm_head = nn.Linear(d_model, vocab_size)          # next-token
        self.concept_head = nn.Linear(d_model, num_concepts)   # classification concept
        # embeddings des concepts (pour L_align)
        self.concept_emb = nn.Embedding(num_concepts, d_model)

    def forward(self, x):
        """
        x: LongTensor [B, T]
        returns:
          lm_logits: [B, T, V]
          concept_logits: [B, C]   (pooling simple: prend état final)
          pooled: [B, H]           (représentation utilisée pour concept)
        """
        h = self.embed(x)                 # [B, T, H]
        out, hT = self.rnn(h)             # out: [B,T,H], hT: [1,B,H]
        lm_logits = self.lm_head(out)     # [B,T,V]
        pooled = hT.squeeze(0)            # [B,H] (dernier état)
        concept_logits = self.concept_head(pooled)  # [B,C]
        return lm_logits, concept_logits, pooled

# ---------------------------------------------------
# 4) Pertes : L_LM + λ1 * L_dist + λ2 * L_align
# ---------------------------------------------------
def language_model_loss(lm_logits, target_tokens, pad_id=0):
    """
    LM teacher forcing simple:
    - On prédit x_{t+1} à partir de x_{<=t}
    - target est x décalé d'un pas
    """
    B, T, V = lm_logits.shape
    # Shift: input[ : , :-1] -> pred ; target -> x[:, 1:]
    pred = lm_logits[:, :-1, :].contiguous().view(-1, V)
    tgt = target_tokens[:, 1:].contiguous().view(-1)
    loss = F.cross_entropy(pred, tgt, ignore_index=pad_id)
    return loss

def ontology_distance_loss(concept_logits, gold_ids, D):
    """
    L_dist = E_{p_hat} [ D[j, gold] ] = sum_j softmax(logits)[j] * D[j,gold]
    """
    probs = F.softmax(concept_logits, dim=-1)      # [B, C]
    B = concept_logits.size(0)
    dist_vals = []
    for b in range(B):
        g = gold_ids[b].item()
        d_row = D[:, g]                            # [C]
        dist_vals.append(torch.sum(probs[b] * d_row))
    return torch.stack(dist_vals).mean()

def ontology_align_loss(pooled_repr, concept_emb, gold_ids):
    """
    L_align = 1 - cos( W*h , E[gold] )
    Ici W*h est déjà 'pooled_repr' (ou on pourrait ajouter une couche)
    """
    gold_vec = concept_emb(gold_ids)               # [B, H]
    cos = F.cosine_similarity(pooled_repr, gold_vec, dim=-1)
    return (1.0 - cos).mean()

# ---------------------------------------------------
# 5) Démo d’un batch : données + entraînement
# ---------------------------------------------------
def demo_batch():
    """
    Exemples médicaux (texte -> concept feuille)
    NB : en production on obtient le gold concept par annotation (SNOMED/UMLS),
         ici on l'écrit à la main pour la démo.
    """
    samples = [
        # CT montre lésion lobe supérieur du poumon → Lesion (UpperLobe)
        ("CT scan shows lesion in upper lobe lung",
         "Body.RespiratorySystem.Lung.Lobe.UpperLobe"),  # noeud interne…
        # …mais notre classification porte sur les FEUILLES. On mappe vers 'Disease' feuille la plus proche.
        # Pour la démo, on décide d'étiqueter comme 'Nodule' (feuille) si 'lesion' (proche sémantiquement).
        # Tu peux changer selon ton schéma.
        # On ajoute donc un 'gold_leaf' à côté :
    ]
    # Pour la clarté, on crée une liste directement au format (text, gold_leaf)
    batch = [
        ("CT scan shows lesion in upper lobe lung",
         "Body.RespiratorySystem.Lung.Disease.Nodule"),
        ("Echocardiogram shows severe mitral valve regurgitation",
         "Body.CardiovascularSystem.Heart.Valve.MitralValve.Regurgitation"),
        ("Lower lobe pneumonia in left lung",
         "Body.RespiratorySystem.Lung.Disease.Pneumonia"),
        ("Aortic valve stenosis suspected",
         "Body.CardiovascularSystem.Heart.Valve.AorticValve.Stenosis"),
        ("Stomach ulcer observed",
         "Body.DigestiveSystem.Stomach.Disease.Ulcer")
    ]
    return batch

def collate_batch(samples, max_len=24):
    xs, ys = [], []
    for text, leaf in samples:
        xs.append(encode(text, max_len))
        ys.append(LEAF2ID[leaf])
    X = torch.stack(xs, dim=0)              # [B,T]
    y = torch.tensor(ys, dtype=torch.long)  # [B]
    return X, y

def main_train_step():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(VOCAB)
    model = MiniOntoLM(vocab_size=vocab_size, d_model=128, num_concepts=NUM_CONCEPTS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # hyper des pertes
    lambda_dist = 1.0
    lambda_align = 0.5

    batch = demo_batch()
    X, gold_concepts = collate_batch(batch, max_len=24)
    X = X.to(device)
    gold_concepts = gold_concepts.to(device)
    Dmat = D.to(device)

    model.train()
    for step in range(300):  # quelques itérations pour la démo
        lm_logits, concept_logits, pooled = model(X)
        # L_LM (next-token)
        L_lm = language_model_loss(lm_logits, X, pad_id=VOCAB["<pad>"])
        # L_dist (distance attendue)
        L_dist = ontology_distance_loss(concept_logits, gold_concepts, Dmat)
        # L_align (cosine)
        L_align = ontology_align_loss(pooled, model.concept_emb, gold_concepts)

        L_total = L_lm + lambda_dist * L_dist + lambda_align * L_align

        opt.zero_grad()
        L_total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            with torch.no_grad():
                pred_id = concept_logits.argmax(dim=-1)
                acc = (pred_id == gold_concepts).float().mean().item()
            print(f"[step {step:03d}] L_lm={L_lm:.3f}  L_dist={L_dist:.3f}  L_align={L_align:.3f}  "
                  f"L_total={L_total:.3f}  concept_acc={acc:.2f}")

    # Inference démo
    model.eval()
    test_txt = "CT shows a nodule in right upper lobe of lung"
    x = encode(test_txt, max_len=24).unsqueeze(0).to(device)
    with torch.no_grad():
        lm_logits, concept_logits, pooled = model(x)
        pred = concept_logits.softmax(-1)
        topk = pred.topk(3, dim=-1)
    print("\n[Inference]")
    for rank in range(topk.indices.size(1)):
        cid = topk.indices[0, rank].item()
        print(f"  {rank+1}. {LEAF_LIST[cid]}  (p={topk.values[0, rank].item():.3f})")

if __name__ == "__main__":
    main_train_step()
