"""Aprendizaje NO supervisado (puntos V.b y V.c del informe).

Basado en las diapositivas «1.10 Aprendizaje no supervisado»:

  - Datos sin etiquetas, estandarizados con z-score, distancia euclidiana.
  - Agrupamiento por particiones  -> K-Means (modelo principal)
        * k elegido con el método del codo y el coeficiente silhouette
        * n_init: varias inicializaciones para evitar óptimos locales
  - Clustering jerárquico aglomerativo (AGNES) -> dendrogramas comparando las
        distancias entre clusters vistas en clase: MIN (single-link),
        MAX (complete-link), Promedio y Centroides.
  - Métodos basados en densidad -> DBSCAN (detecta ruido / outliers,
        que es la desventaja de K-Means). eps se elige con el gráfico k-distancia.

Las etiquetas reales NO se usan para agrupar; solo al final para evaluar
(pureza, ARI, NMI) y para nombrar cada cluster con el estudiante mayoritario.

Uso:
    python scripts/03_clustering.py
    python scripts/03_clustering.py --k 12     # forzar k
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
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, adjusted_rand_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src import config as C

ENLACES = {   # nombre en las diapositivas -> método de scipy
    "MIN (single-link)": "single",
    "MAX (complete-link)": "complete",
    "Promedio": "average",
    "Centroides": "centroid",
}
MAX_PUNTOS_JERARQUICO = 2000   # el jerárquico es O(n²): baja escalabilidad


def pureza(y_real, y_cluster):
    cm = contingency_matrix(y_real, y_cluster)
    return cm.max(axis=0).sum() / cm.sum()


def silueta(Z, etiquetas):
    n = len(set(etiquetas))
    return silhouette_score(Z, etiquetas) if 1 < n < len(Z) else float("nan")


def evaluar(nombre, Z, y, etiquetas, ruido=0.0):
    return {"algoritmo": nombre,
            "clusters": len(set(etiquetas) - {-1}),
            "silhouette": round(silueta(Z, etiquetas), 4),
            "ARI": round(adjusted_rand_score(y, etiquetas), 4),
            "NMI": round(normalized_mutual_info_score(y, etiquetas), 4),
            "pureza": round(float(pureza(y, etiquetas)), 4),
            "ruido": round(ruido, 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=None, help="por defecto = nº de estudiantes")
    a = p.parse_args()

    d = np.load(C.DATA_PROCESSED / "dataset.npz", allow_pickle=True)
    clases = list(d["clases"])
    entren = d["split"] == "train"
    test = d["split"] == "test"
    X_tr, y_tr = d["X_vec"][entren], d["y"][entren]
    X_te, y_te = d["X_vec"][test], d["y"][test]
    C.REPORTS.mkdir(exist_ok=True)
    C.MODELS.mkdir(exist_ok=True)
    print(f"{len(X_tr)} segmentos para agrupar, {len(X_te)} para probar, "
          f"{len(clases)} estudiantes, {X_tr.shape[1]} características MFCC")

    # --- Estandarización z-score -------------------------------------------------
    scaler = StandardScaler().fit(X_tr)
    Z_tr, Z_te = scaler.transform(X_tr), scaler.transform(X_te)
    k = a.k or len(clases)
    filas = []

    # =========================================================================
    # 1. K-MEANS (agrupamiento por particiones)
    # =========================================================================
    print("\n1) K-Means: buscando k con el método del codo y silhouette...")
    ks = list(range(2, max(min(len(clases) * 2 + 2, len(Z_tr) - 1, 40), 3) + 1))
    inercias, siluetas = [], []
    for kk in ks:
        km_k = KMeans(n_clusters=kk, n_init=10, random_state=C.SEMILLA).fit(Z_tr)
        inercias.append(km_k.inertia_)
        siluetas.append(silhouette_score(Z_tr, km_k.labels_))
    k_silueta = ks[int(np.argmax(siluetas))]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ks, inercias, "o-"); ax[0].set_title("Método del codo"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inercia (suma de distancias²)")
    ax[1].plot(ks, siluetas, "o-"); ax[1].set_title("Coeficiente silhouette"); ax[1].set_xlabel("k")
    for e in ax:
        e.axvline(len(clases), color="red", ls="--", label=f"nº estudiantes = {len(clases)}")
        e.legend()
    fig.tight_layout(); fig.savefig(C.REPORTS / "kmeans_codo_silhouette.png", dpi=150); plt.close(fig)
    print(f"   k sugerido por silhouette = {k_silueta} (estudiantes = {len(clases)})")

    print(f"   Entrenando K-Means final con k={k} (20 inicializaciones)...")
    km = KMeans(n_clusters=k, n_init=20, random_state=C.SEMILLA).fit(Z_tr)
    filas.append(evaluar("K-Means", Z_tr, y_tr, km.labels_))

    # Nombrar cada cluster con el estudiante mayoritario (para usarlo en la asistencia)
    mapa = {}
    for c in range(k):
        miembros = y_tr[km.labels_ == c]
        mapa[c] = int(np.bincount(miembros).argmax()) if len(miembros) else -1
    dist_tr = km.transform(Z_tr).min(axis=1)
    umbral = float(np.percentile(dist_tr, C.PERCENTIL_DESCONOCIDO) * C.FACTOR_UMBRAL)
    if len(Z_te):
        pred_te = np.array([mapa[c] for c in km.predict(Z_te)])
        acc_te = accuracy_score(y_te, pred_te)
    else:
        acc_te = float("nan")
    print(f"   {filas[-1]}")
    print(f"   accuracy en test (cluster -> estudiante) = {acc_te:.3f}")

    # =========================================================================
    # 2. CLUSTERING JERÁRQUICO AGLOMERATIVO (AGNES)
    # =========================================================================
    print("\n2) Clustering jerárquico: comparando MIN, MAX, Promedio y Centroides...")
    rng = np.random.default_rng(C.SEMILLA)
    idx_h = rng.choice(len(Z_tr), size=min(MAX_PUNTOS_JERARQUICO, len(Z_tr)), replace=False)
    Zh, yh = Z_tr[idx_h], y_tr[idx_h]
    etiquetas_h = {}
    for nombre, metodo in ENLACES.items():
        L = linkage(Zh, method=metodo)
        et = fcluster(L, t=k, criterion="maxclust")
        etiquetas_h[nombre] = et
        filas.append(evaluar(f"Jerárquico {nombre}", Zh, yh, et))
        print(f"   {filas[-1]}")

    # Dendrogramas (sobre una muestra pequeña para que se puedan leer)
    sub = rng.choice(len(Z_tr), size=min(120, len(Z_tr)), replace=False)
    nombres_sub = [clases[i] for i in y_tr[sub]]
    fig, axs = plt.subplots(2, 2, figsize=(20, 11))
    for e, (nombre, metodo) in zip(axs.ravel(), ENLACES.items()):
        L = linkage(Z_tr[sub], method=metodo)
        corte = L[-(k - 1), 2] if 1 < k <= len(L) else None
        dendrogram(L, labels=nombres_sub, leaf_rotation=90, leaf_font_size=6, ax=e,
                   color_threshold=corte)
        e.set_title(f"Enlace {nombre}")
        e.set_ylabel("distancia")
    fig.suptitle("Dendrogramas — clustering jerárquico aglomerativo", fontsize=14)
    fig.tight_layout(); fig.savefig(C.REPORTS / "jerarquico_dendrograma.png", dpi=150); plt.close(fig)

    # =========================================================================
    # 3. DBSCAN (basado en densidad)
    # =========================================================================
    print("\n3) DBSCAN: eligiendo eps con el gráfico k-distancia...")
    min_samples = 5
    nn = NearestNeighbors(n_neighbors=min_samples).fit(Z_tr)
    kdist = np.sort(nn.kneighbors(Z_tr)[0][:, -1])
    eps = float(np.percentile(kdist, 90))
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(Z_tr)
    ruido = float(np.mean(db.labels_ == -1))
    filas.append(evaluar("DBSCAN", Z_tr, y_tr, db.labels_, ruido))
    print(f"   eps={eps:.2f} | {filas[-1]}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(kdist); ax.axhline(eps, color="red", ls="--", label=f"eps elegido = {eps:.2f}")
    ax.set_title(f"Gráfico k-distancia (k = min_samples = {min_samples})")
    ax.set_xlabel("puntos ordenados"); ax.set_ylabel(f"distancia al {min_samples}º vecino"); ax.legend()
    fig.tight_layout(); fig.savefig(C.REPORTS / "dbscan_k_distancia.png", dpi=150); plt.close(fig)

    # =========================================================================
    # 4. Visualización 2D (PCA) y comparación
    # =========================================================================
    tabla = pd.DataFrame(filas).set_index("algoritmo")
    jer = tabla.filter(like="Jerárquico", axis=0)
    mejor_jer = jer["ARI"].idxmax()

    P = PCA(n_components=2, random_state=C.SEMILLA).fit_transform(Z_tr)
    Ph = P[idx_h]
    fig, axs = plt.subplots(2, 2, figsize=(14, 11))
    vistas = [(P, y_tr, "Etiqueta real (solo como referencia)"),
              (P, km.labels_, f"K-Means (k={k})"),
              (Ph, etiquetas_h[mejor_jer.replace("Jerárquico ", "")], mejor_jer),
              (P, db.labels_, "DBSCAN (gris = ruido)")]
    for e, (pts, col, tit) in zip(axs.ravel(), vistas):
        col = np.asarray(col)
        if "DBSCAN" in tit:
            e.scatter(pts[col == -1, 0], pts[col == -1, 1], c="lightgray", s=8, label="ruido")
            pts, col = pts[col != -1], col[col != -1]
        e.scatter(pts[:, 0], pts[:, 1], c=col, cmap="tab20", s=10)
        e.set_title(tit); e.set_xlabel("PC1"); e.set_ylabel("PC2")
    fig.tight_layout(); fig.savefig(C.REPORTS / "clusters_pca.png", dpi=150); plt.close(fig)

    tabla.to_csv(C.REPORTS / "comparacion_no_supervisado.csv")
    print("\nComparación de algoritmos no supervisados:")
    print(tabla.to_string())

    # --- Guardar -------------------------------------------------------------------
    joblib.dump({"scaler": scaler, "kmeans": km, "mapa": mapa, "umbral": umbral,
                 "clases": clases}, C.MODELS / "clustering.joblib")
    fk = tabla.loc["K-Means"]
    (C.REPORTS / "metricas_kmeans.json").write_text(json.dumps({
        "modelo": "K-MEANS",
        "k": k,
        "k_sugerido_silhouette": k_silueta,
        "silhouette": float(fk["silhouette"]),
        "ARI": float(fk["ARI"]),
        "NMI": float(fk["NMI"]),
        "pureza": float(fk["pureza"]),
        "accuracy_test": round(acc_te, 4),
        "mejor_jerarquico": mejor_jer,
        "dbscan_eps": round(eps, 4),
        "dbscan_clusters": int(tabla.loc["DBSCAN", "clusters"]),
        "dbscan_ruido": round(ruido, 4),
        "requiere_etiquetas": False,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nModelo guardado en models/clustering.joblib | gráficas y tabla en reports/")


if __name__ == "__main__":
    main()
