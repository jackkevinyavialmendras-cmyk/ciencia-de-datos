"""Modelo 2 — Aprendizaje no supervisado (puntos V.b y V.c del informe).

Sobre vectores MFCC (media + desviación) estandarizados con z-score:
  1. K-Means      : modelo principal; k elegido con codo + silhouette.
  2. Jerárquico   : dendrograma aglomerativo (AGNES, enlace Ward) para validar
                    el número de grupos y ver qué voces se parecen.
  3. DBSCAN       : detecta ruido / outliers (voces que no encajan en ningún grupo).

Las etiquetas reales NO se usan para entrenar; solo para evaluar (pureza, ARI, NMI)
y para nombrar cada cluster con el estudiante mayoritario.

Uso:
    python scripts/04_clustering.py
    python scripts/04_clustering.py --k 12     # forzar k
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, adjusted_rand_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src import config as C


def pureza(y_real, y_cluster):
    cm = contingency_matrix(y_real, y_cluster)
    return cm.max(axis=0).sum() / cm.sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=None, help="por defecto = nº de estudiantes")
    a = p.parse_args()

    d = np.load(C.DATA_PROCESSED / "dataset.npz", allow_pickle=True)
    clases = list(d["clases"])
    # Sin segmentos aumentados: el clustering se ajusta a voces reales
    entren = np.isin(d["split"], ["train", "val"]) & ~d["aumentado"]
    test = d["split"] == "test"
    X_tr, y_tr = d["X_vec"][entren], d["y"][entren]
    X_te, y_te = d["X_vec"][test], d["y"][test]
    C.REPORTS.mkdir(exist_ok=True)
    C.MODELS.mkdir(exist_ok=True)

    # --- z-score ------------------------------------------------------------
    scaler = StandardScaler().fit(X_tr)
    Z_tr, Z_te = scaler.transform(X_tr), scaler.transform(X_te)

    # --- 1. Elegir k: codo + silhouette ---------------------------------------
    print("1) Buscando k (codo y silhouette)...")
    k_max = min(len(clases) * 2 + 2, len(Z_tr) - 1, 40)
    ks = list(range(2, max(k_max, 3) + 1))
    inercias, siluetas = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=C.SEMILLA).fit(Z_tr)
        inercias.append(km.inertia_)
        siluetas.append(silhouette_score(Z_tr, km.labels_))
    k_silueta = ks[int(np.argmax(siluetas))]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ks, inercias, "o-"); ax[0].set_title("Método del codo"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inercia")
    ax[1].plot(ks, siluetas, "o-"); ax[1].set_title("Coeficiente silhouette"); ax[1].set_xlabel("k")
    for e in ax:
        e.axvline(len(clases), color="red", ls="--", label=f"nº estudiantes = {len(clases)}")
        e.legend()
    fig.tight_layout(); fig.savefig(C.REPORTS / "kmeans_codo_silhouette.png", dpi=150)
    print(f"   mejor k por silhouette = {k_silueta} (estudiantes = {len(clases)})")

    # --- 2. K-Means final -----------------------------------------------------
    k = a.k or len(clases)
    print(f"2) Entrenando K-Means con k={k}...")
    km = KMeans(n_clusters=k, n_init=20, random_state=C.SEMILLA).fit(Z_tr)
    etiquetas = km.labels_
    sil = silhouette_score(Z_tr, etiquetas)
    ari = adjusted_rand_score(y_tr, etiquetas)
    nmi = normalized_mutual_info_score(y_tr, etiquetas)
    pur = pureza(y_tr, etiquetas)

    # Nombrar cada cluster con el estudiante mayoritario (solo para usarlo en la app)
    mapa = {}
    for c in range(k):
        miembros = y_tr[etiquetas == c]
        mapa[c] = int(np.bincount(miembros).argmax()) if len(miembros) else -1
    dist_tr = km.transform(Z_tr).min(axis=1)
    umbral = float(np.percentile(dist_tr, C.PERCENTIL_DESCONOCIDO))

    pred_te = np.array([mapa[c] for c in km.predict(Z_te)]) if len(Z_te) else np.array([])
    acc_te = accuracy_score(y_te, pred_te) if len(Z_te) else float("nan")
    print(f"   silhouette={sil:.3f} | ARI={ari:.3f} | NMI={nmi:.3f} | pureza={pur:.3f}")
    print(f"   accuracy en test (cluster -> estudiante) = {acc_te:.3f}")

    # --- 3. Jerárquico: dendrograma -------------------------------------------
    print("3) Clustering jerárquico (dendrograma)...")
    rng = np.random.default_rng(C.SEMILLA)
    sub = rng.choice(len(Z_tr), size=min(150, len(Z_tr)), replace=False)
    Lk = linkage(Z_tr[sub], method="ward")
    fig, ax = plt.subplots(figsize=(16, 6))
    dendrogram(Lk, labels=[clases[i] for i in y_tr[sub]], leaf_rotation=90,
               leaf_font_size=7, ax=ax, color_threshold=Lk[-(len(clases) - 1), 2] if len(clases) > 1 else None)
    ax.set_title("Dendrograma (aglomerativo, enlace Ward)")
    fig.tight_layout(); fig.savefig(C.REPORTS / "jerarquico_dendrograma.png", dpi=150)

    # --- 4. DBSCAN: ruido / outliers ------------------------------------------
    print("4) DBSCAN (detección de ruido)...")
    min_samples = 5
    nn = NearestNeighbors(n_neighbors=min_samples).fit(Z_tr)
    kdist = np.sort(nn.kneighbors(Z_tr)[0][:, -1])
    eps = float(np.percentile(kdist, 90))
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(Z_tr)
    n_db = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    ruido = float(np.mean(db.labels_ == -1))
    print(f"   eps={eps:.2f} | clusters={n_db} | ruido={ruido:.1%}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(kdist); ax.axhline(eps, color="red", ls="--", label=f"eps = {eps:.2f}")
    ax.set_title(f"Gráfico k-distancia (k={min_samples})"); ax.set_xlabel("puntos ordenados"); ax.legend()
    fig.tight_layout(); fig.savefig(C.REPORTS / "dbscan_k_distancia.png", dpi=150)

    # --- 5. Visualización 2D con PCA -----------------------------------------
    P = PCA(n_components=2, random_state=C.SEMILLA).fit_transform(Z_tr)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for e, col, tit in zip(ax, [y_tr, etiquetas, db.labels_],
                           ["Etiqueta real (solo referencia)", f"K-Means (k={k})", "DBSCAN (-1 = ruido)"]):
        e.scatter(P[:, 0], P[:, 1], c=col, cmap="tab20", s=10)
        e.set_title(tit); e.set_xlabel("PC1"); e.set_ylabel("PC2")
    fig.tight_layout(); fig.savefig(C.REPORTS / "clusters_pca.png", dpi=150)

    # --- Guardar ---------------------------------------------------------------
    joblib.dump({"scaler": scaler, "kmeans": km, "mapa": mapa, "umbral": umbral,
                 "clases": clases}, C.MODELS / "clustering.joblib")
    (C.REPORTS / "metricas_kmeans.json").write_text(json.dumps({
        "modelo": "K-MEANS",
        "k": k,
        "k_sugerido_silhouette": k_silueta,
        "silhouette": round(sil, 4),
        "ARI": round(ari, 4),
        "NMI": round(nmi, 4),
        "pureza": round(pur, 4),
        "accuracy_test": round(acc_te, 4),
        "dbscan_eps": round(eps, 4),
        "dbscan_clusters": n_db,
        "dbscan_ruido": round(ruido, 4),
        "requiere_etiquetas": False,
    }, indent=2), encoding="utf-8")
    print("\nModelo guardado en models/clustering.joblib | gráficas en reports/")


if __name__ == "__main__":
    main()
