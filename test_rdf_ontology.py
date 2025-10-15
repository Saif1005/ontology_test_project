
"""

Pipeline orienté-classes pour prétraiter une ontologie RDF/OWL :
- Chargement RDF/OWL
- Hiérarchie (rdfs:subClassOf + optionnel OWL Restriction part_of)
- Racines/Feuilles
- Chemins canoniques + dot-notation
- Matrice des distances D (plus court chemin, normalisée 0..1)
- Exports artefacts


"""

from __future__ import annotations
import os, json, math, argparse
from dataclasses import dataclass
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict, deque

import numpy as np
from rdflib import Graph, Namespace, URIRef, RDF, RDFS, OWL, Literal


SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
OBO  = Namespace("http://purl.obolibrary.org/obo/")
OIO  = Namespace("http://www.geneontology.org/formats/oboInOwl#")

# Propriété "part_of" (BFO_0000050) si on souhaite l'inclure via Restriction OWL
BFO_PART_OF = URIRef("http://purl.obolibrary.org/obo/BFO_0000050")


@dataclass
class OntologyConfig:
    rdf_path: str
    out_dir: str = "./onto_artifacts"
    iri_filter: str = ""                 # garder uniquement les classes dont l'IRI contient ce motif (ex: "/SO_")
    include_part_of: bool = False        # inclure part_of via OWL Restriction
    restrict_contains: str = ""          # ne garder que les feuilles dont le chemin contient ce token
    prefer_roots_with: str = ""          # favoriser racines contenant ce token (pour choix du chemin canonique)
    max_predicates_preview: int = 0      # 0 = pas d'aperçu ; sinon imprime N prédicats uniques vus


class OntologyGraphLoader:
    """Charge le graphe RDF/OWL et fournit utilitaires de labelisation."""

    def __init__(self, cfg: OntologyConfig):
        self.cfg = cfg
        self.g = Graph()

    def load(self) -> Graph:
        # rdflib détecte souvent bien le format ; sinon, ajoute format="xml" si c'est OWL/XML
        self.g.parse(self.cfg.rdf_path)
        return self.g

    def label(self, c: URIRef) -> str:
        # rdfs:label > skos:prefLabel > fragment d'URI
        for p in (RDFS.label, SKOS.prefLabel):
            for obj in self.g.objects(c, p):
                if isinstance(obj, Literal):
                    return str(obj)
        s = str(c)
        return s.split("#")[-1] if "#" in s else s.rsplit("/", 1)[-1]

    def synonyms(self, c: URIRef) -> List[str]:
        syns = set()
        for p in (OIO.hasExactSynonym, OIO.hasRelatedSynonym, OIO.hasBroadSynonym, OIO.hasNarrowSynonym):
            for obj in self.g.objects(c, p):
                if isinstance(obj, Literal):
                    syns.add(str(obj))
        return sorted(syns)

    def preview(self):
        print("== Namespaces ==")
        for prefix, uri in self.g.namespaces():
            print(f"{prefix}: {uri}")
        if self.cfg.max_predicates_preview > 0:
            print("\n== Predicates (sample) ==")
            seen = set()
            for _, p, _ in self.g.triples((None, None, None)):
                if p not in seen:
                    seen.add(p); print(p)
                if len(seen) >= self.cfg.max_predicates_preview:
                    break


class OntologyHierarchyBuilder:
    """Construit parents/enfants depuis rdfs:subClassOf (+ optionnel BFO:part_of via OWL Restriction)."""

    def __init__(self, g: Graph, iri_filter: str = "", include_part_of: bool = False):
        self.g = g
        self.iri_filter = iri_filter
        self.include_part_of = include_part_of
        self.classes: Set[URIRef] = set()
        self.parents: Dict[URIRef, Set[URIRef]] = defaultdict(set)
        self.children: Dict[URIRef, Set[URIRef]] = defaultdict(set)

    def _collect_classes(self):
        all_classes = set(self.g.subjects(RDF.type, OWL.Class)) | set(self.g.subjects(RDF.type, RDFS.Class))
        all_classes = {c for c in all_classes if isinstance(c, URIRef)}
        if self.iri_filter:
            all_classes = {c for c in all_classes if self.iri_filter in str(c)}
        self.classes = all_classes

    def _add_edge(self, child: URIRef, parent: URIRef):
        if child in self.classes and parent in self.classes:
            self.parents[child].add(parent)
            self.children[parent].add(child)

    def build(self) -> Tuple[Set[URIRef], Dict[URIRef, Set[URIRef]], Dict[URIRef, Set[URIRef]]]:
        self._collect_classes()

        # 2.1 subClassOf directs
        for c in self.classes:
            for p in self.g.objects(c, RDFS.subClassOf):
                if isinstance(p, URIRef) and p in self.classes:
                    self._add_edge(c, p)

        # 2.2 OWL Restriction pour part_of (optionnel)
        if self.include_part_of:
            for c in self.classes:
                for sup in self.g.objects(c, RDFS.subClassOf):
                    if (sup, RDF.type, OWL.Restriction) in self.g:
                        on_prop = self.g.value(subject=sup, predicate=OWL.onProperty)
                        if on_prop == BFO_PART_OF:
                            filler = self.g.value(subject=sup, predicate=OWL.someValuesFrom)
                            if isinstance(filler, URIRef) and filler in self.classes:
                                self._add_edge(c, filler)

        return self.classes, self.parents, self.children

    @staticmethod
    def find_roots(classes: Set[URIRef], parents: Dict[URIRef, Set[URIRef]]) -> List[URIRef]:
        return [c for c in classes if len(parents.get(c, set())) == 0]

    @staticmethod
    def find_leaves(classes: Set[URIRef], children: Dict[URIRef, Set[URIRef]]) -> List[URIRef]:
        return [c for c in classes if len(children.get(c, set())) == 0]


class PathCanonicalizer:
    """Génère tous les chemins leaf->root et choisit un chemin canonique par feuille."""

    def __init__(self, g: Graph, label_fn, parents: Dict[URIRef, Set[URIRef]]):
        self.g = g
        self.parents = parents
        self.label = label_fn

    def all_paths_to_roots(self, node: URIRef) -> List[List[URIRef]]:
        ps = self.parents.get(node, set())
        if not ps:
            return [[node]]
        out = []
        for p in ps:
            for up in self.all_paths_to_roots(p):
                out.append([node] + up)
        return out

    def canonical_path(self, leaf: URIRef, preferred_roots: Optional[Set[URIRef]] = None) -> List[URIRef]:
        cands = self.all_paths_to_roots(leaf)
        if preferred_roots:
            filt = [p for p in cands if p and p[-1] in preferred_roots]
            if filt:
                cands = filt
        # tri: plus court, puis lexicographique sur labels (du root vers leaf)
        def keyf(p):
            labs = [self.label(n) for n in p]
            return (len(p), " > ".join(reversed(labs)))
        cands.sort(key=keyf)
        return cands[0] if cands else [leaf]

    def path_to_dot(self, path_root_to_leaf: List[URIRef]) -> str:
        parts = [(self.label(n) or "").replace(".", "_").strip() for n in path_root_to_leaf]
        return ".".join(parts)


class DistanceComputer:
    """Construit l’adjacence non orientée et calcule la matrice D (plus court chemin, normalisée 0..1)."""

    def __init__(self, parents: Dict[URIRef, Set[URIRef]], children: Dict[URIRef, Set[URIRef]]):
        self.parents = parents
        self.children = children
        self.adj: Dict[URIRef, Set[URIRef]] = defaultdict(set)

    def build_adj(self):
        for c, ps in self.parents.items():
            for p in ps:
                self.adj[c].add(p); self.adj[p].add(c)
        for p, cs in self.children.items():
            for c in cs:
                self.adj[p].add(c); self.adj[c].add(p)

    def shortest_path_len(self, s: URIRef, t: URIRef) -> int:
        if s == t:
            return 0
        q = deque([s]); dist = {s: 0}
        while q:
            x = q.popleft()
            for y in self.adj.get(x, set()):
                if y not in dist:
                    dist[y] = dist[x] + 1
                    if y == t:
                        return dist[y]
                    q.append(y)
        return math.inf

    def compute_D(self, leaves_uri: List[URIRef]) -> np.ndarray:
        self.build_adj()
        n = len(leaves_uri)
        D = np.zeros((n, n), dtype=np.float32)
        maxd = 1
        for i, a in enumerate(leaves_uri):
            for j, b in enumerate(leaves_uri):
                d = self.shortest_path_len(a, b)
                if d != math.inf:
                    D[i, j] = d
                    if d > maxd: maxd = d
                else:
                    D[i, j] = maxd
        if maxd > 0:
            D = D / maxd
        return D


class ArtifactExporter:
    """Sauvegarde les artefacts (json / npy) et affiche quelques stats."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def save_json(self, name: str, obj):
        with open(os.path.join(self.out_dir, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def save_npy(self, name: str, arr: np.ndarray):
        np.save(os.path.join(self.out_dir, name), arr)

    def log_summary(self, leaf_list: List[str], roots_dot: List[str], D: np.ndarray):
        print(f"✅ Export OK → {self.out_dir}")
        print(f"   • feuilles retenues: {len(leaf_list)}")
        print(f"   • racines détectées: {len(roots_dot)}")
        print(f"   • D shape: {tuple(D.shape)}, min={float(D.min()):.2f}, max={float(D.max()):.2f}")
        print(f"   • exemples LEAF_LIST: {leaf_list[:5]}")


class OntologyPreprocessPipeline:
    """Pipeline complet : load → hierarchy → paths → dots → D → export."""

    def __init__(self, cfg: OntologyConfig):
        self.cfg = cfg
        self.loader = OntologyGraphLoader(cfg)

    def run(self):
        g = self.loader.load()
        if self.cfg.max_predicates_preview:
            self.loader.preview()

        # Hiérarchie
        hier = OntologyHierarchyBuilder(
            g=g,
            iri_filter=self.cfg.iri_filter,
            include_part_of=self.cfg.include_part_of
        )
        classes, parents, children = hier.build()
        roots = OntologyHierarchyBuilder.find_roots(classes, parents)
        leaves = OntologyHierarchyBuilder.find_leaves(classes, children)

        # Préférence de racines
        preferred_roots: Optional[Set[URIRef]] = None
        if self.cfg.prefer_roots_with:
            token = self.cfg.prefer_roots_with.lower()
            preferred_roots = {r for r in roots if token in self.loader.label(r).lower()}

        # Chemins canoniques
        canon = PathCanonicalizer(g, self.loader.label, parents)
        leaf_paths_uri: List[List[URIRef]] = []
        leaf_nodes_uri: List[URIRef] = []
        token_filter = (self.cfg.restrict_contains or "").lower()

        for lf in leaves:
            path_lr = list(reversed(canon.canonical_path(lf, preferred_roots)))  # root->...->leaf
            if token_filter:
                labs = [self.loader.label(n).lower() for n in path_lr]
                if not any(token_filter in lab for lab in labs):
                    continue
            leaf_paths_uri.append(path_lr)
            leaf_nodes_uri.append(lf)

        # Dot-notation & mappings
        dot_paths = [canon.path_to_dot(p) for p in leaf_paths_uri]
        seen = set()
        leaf_list: List[str] = []
        for p in dot_paths:
            if p not in seen:
                seen.add(p); leaf_list.append(p)

        leaf2id = {c: i for i, c in enumerate(leaf_list)}
        id2leaf = {i: c for c, i in leaf2id.items()}

        # Distances D
        dist = DistanceComputer(parents, children)
        D = dist.compute_D(leaf_nodes_uri)  # même ordre que leaf_nodes_uri (OK pour entraînement interne)

        # Racines (dot) pour diagnostic
        roots_dot = [canon.path_to_dot([r]) for r in roots]

        # Exports
        exp = ArtifactExporter(self.cfg.out_dir)
        exp.save_json("LEAF_LIST.json", leaf_list)
        exp.save_json("LEAF2ID.json", leaf2id)
        exp.save_json("ID2LEAF.json", id2leaf)
        exp.save_json("roots.json", roots_dot)
        exp.save_npy("D.npy", D)
        exp.log_summary(leaf_list, roots_dot, D)



def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdf", required=True, help="Chemin du fichier RDF/OWL/TTL")
    ap.add_argument("--out_dir", default="./onto_artifacts", help="Dossier de sortie")
    ap.add_argument("--iri_filter", default="", help="Garder classes dont l'IRI contient ce motif (ex: '/SO_')")
    ap.add_argument("--include_part_of", action="store_true", help="Inclure part_of via OWL Restriction (BFO_0000050)")
    ap.add_argument("--restrict_contains", default="", help="Ne garder que les feuilles dont le chemin contient ce token")
    ap.add_argument("--prefer_roots_with", default="", help="Favoriser chemins passant par une racine contenant ce token")
    ap.add_argument("--preview_predicates", type=int, default=0, help="Aperçu N prédicats distincts")
    return ap.parse_args()

def main():
    args = build_args()
    cfg = OntologyConfig(
        rdf_path=args.rdf,
        out_dir=args.out_dir,
        iri_filter=args.iri_filter,
        include_part_of=args.include_part_of,
        restrict_contains=args.restrict_contains,
        prefer_roots_with=args.prefer_roots_with,
        max_predicates_preview=args.preview_predicates,
    )
    OntologyPreprocessPipeline(cfg).run()

if __name__ == "__main__":
    main()
