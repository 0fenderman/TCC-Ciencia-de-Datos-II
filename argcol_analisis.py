# -*- coding: utf-8 -*-
"""
TCC - Ciencia de Datos II (Universidad de Cartagena, C.T. Lorica)
Caso ARGCOL - tienda de ropa streetwear.
Script integrador del proyecto:
  1) carga, limpieza, estandarización, imputación, outliers, one-hot, características
  2) regresión logística (objetivo: entrega_rapida) + referencia base
  3) K-Means (k=4) sobre pedidos, PCA, estabilidad, chi-cuadrado
Uso:  python argcol_analisis.py      (requiere data/ARGCOL_pedidos_original.csv)
"""
import os, unicodedata, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, silhouette_score, adjusted_rand_score)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency

SEED = 42
os.makedirs("graficas", exist_ok=True); os.makedirs("tablas", exist_ok=True)
sns.set_style("whitegrid")
R = {}   # resultados clave

# ============================ PREPARACIÓN Y LIMPIEZA ============================
df = pd.read_csv("data/ARGCOL_pedidos_original.csv")
R["filas_original"], R["cols_original"] = df.shape
R["nulos_original"] = df.isnull().sum()[df.isnull().sum() > 0].to_dict()
cols_sin_id = [c for c in df.columns if c != "id_pedido"]
R["duplicados"] = int(df.duplicated(subset=cols_sin_id).sum())
R["variantes_ciudad_original"] = int(df["ciudad"].dropna().nunique())
R["calif_fuera_rango"] = int(((df.calificacion < 1) | (df.calificacion > 5)).sum())
R["precio_negativo"] = int((df.precio_unitario < 0).sum())
df_raw = df.copy()
df = df.drop_duplicates(subset=cols_sin_id, keep="first").reset_index(drop=True)

def norm(t):
    if pd.isna(t): return t
    t = unicodedata.normalize("NFKD", str(t).strip()).encode("ascii", "ignore").decode()
    return t.lower()
mapas = {
 "ciudad": {"bogota":"Bogota","bogota d.c.":"Bogota","medellin":"Medellin","cartagena":"Cartagena","cali":"Cali","barranquilla":"Barranquilla"},
 "talla": {"s":"S","m":"M","l":"L","large":"L","xl":"XL"},
 "metodo_pago": {"nequi":"Nequi","efectivo":"Efectivo","transferencia":"Transferencia"},
 "canal": {"whatsapp":"WhatsApp","instagram":"Instagram","sitio web":"Sitio web"}}
for c, m in mapas.items():
    df[c] = df[c].apply(norm).map(m)

def parsear_fecha(v):
    if pd.isna(v): return pd.NaT
    for f in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try: return pd.to_datetime(v, format=f)
        except (ValueError, TypeError): continue
    return pd.NaT
df["fecha_pedido"] = df["fecha_pedido"].apply(parsear_fecha)

for c in ["ciudad", "talla", "metodo_pago"]: df[c] = df[c].fillna("No registrado")
df["nombre_cliente"] = df["nombre_cliente"].fillna("Cliente sin nombre")
mediana_calif = df["calificacion"].median()
df["calificacion"] = df["calificacion"].fillna(mediana_calif)
n0 = len(df); df = df.dropna(subset=["fecha_pedido"]).reset_index(drop=True)
R["filas_sin_fecha"] = n0 - len(df)

df.loc[(df.calificacion < 1) | (df.calificacion > 5), "calificacion"] = mediana_calif
neg = df.precio_unitario < 0
df.loc[neg, "precio_unitario"] = df.loc[neg, "precio_unitario"].abs()

def lim_iqr(s):
    q1, q3 = s.quantile(.25), s.quantile(.75); i = q3 - q1
    return q1 - 1.5*i, q3 + 1.5*i
out = {}
for c in ["precio_unitario", "cantidad", "tiempo_entrega_dias"]:
    li, ls = lim_iqr(df[c]); out[c] = {"lim_inf": round(li,1), "lim_sup": round(ls,1),
                                      "n_outliers": int(((df[c]<li)|(df[c]>ls)).sum())}
R["outliers_iqr"] = out
pd.DataFrame(out).T.to_csv("tablas/outliers_iqr.csv")
_, ls_precio = lim_iqr(df.precio_unitario)
df["precio_unitario"] = df.precio_unitario.clip(upper=ls_precio)
df["tiempo_entrega_dias"] = df.tiempo_entrega_dias.clip(upper=10)
df["cantidad"] = df.cantidad.astype(int); df["calificacion"] = df.calificacion.astype(int)

df_clean = df.copy()   # versión limpia (categorías legibles) para EDA
enc = pd.get_dummies(df, columns=["talla","metodo_pago","canal","producto"], prefix=["talla","pago","canal","producto"])
enc["valor_total"] = df.cantidad * df.precio_unitario
enc["mes_pedido"] = df.fecha_pedido.dt.month
enc["dia_semana_pedido"] = df.fecha_pedido.dt.day_name()
enc["pedido_grande"] = (df.cantidad > 1).astype(int)
enc["entrega_rapida"] = (df.tiempo_entrega_dias <= 2).astype(int)
enc.to_csv("data/ARGCOL_pedidos_preparado.csv", index=False, encoding="utf-8")
R["filas_prep"], R["cols_prep"] = enc.shape
R["fecha_min"], R["fecha_max"] = str(df.fecha_pedido.min().date()), str(df.fecha_pedido.max().date())

# --- EDA complementario (figuras del TCC) ---
df_clean["valor_total"] = enc["valor_total"]; df_clean["mes"] = df_clean.fecha_pedido.dt.to_period("M").astype(str)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
df_clean.groupby("mes").size().plot(kind="bar", ax=ax[0], color="#4472C4"); ax[0].set_title("Pedidos por mes"); ax[0].set_xlabel(""); ax[0].set_ylabel("Pedidos")
df_clean.groupby("producto").valor_total.sum().sort_values().plot(kind="barh", ax=ax[1], color="#ED7D31"); ax[1].set_title("Ingresos por producto (COP)"); ax[1].set_ylabel("")
plt.tight_layout(); plt.savefig("graficas/eda_mes_producto.png", dpi=130); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
df_clean.canal.value_counts().plot(kind="bar", ax=ax[0], color="#4472C4"); ax[0].set_title("Pedidos por canal")
df_clean.ciudad.value_counts().plot(kind="bar", ax=ax[1], color="#70AD47"); ax[1].set_title("Pedidos por ciudad")
df_clean.tiempo_entrega_dias.value_counts().sort_index().plot(kind="bar", ax=ax[2], color="#ED7D31"); ax[2].set_title("Días de entrega")
for a in ax: a.set_xlabel(""); a.tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.savefig("graficas/eda_canal_ciudad_entrega.png", dpi=130); plt.close()
R["ingresos_total"] = float(df_clean.valor_total.sum())
R["ingreso_por_producto"] = df_clean.groupby("producto").valor_total.sum().round(0).to_dict()
R["pedidos_por_producto"] = df_clean.producto.value_counts().to_dict()
R["pedidos_por_canal"] = df_clean.canal.value_counts().to_dict()
R["pedidos_por_ciudad"] = df_clean.ciudad.value_counts().to_dict()
R["pedidos_por_mes"] = df_clean.groupby("mes").size().to_dict()
R["calif_media"] = float(df_clean.calificacion.mean()); R["entrega_media"] = float(df_clean.tiempo_entrega_dias.mean())

# ============================ MODELO SUPERVISADO ============================
d = pd.read_csv("data/ARGCOL_pedidos_preparado.csv")
# columnas legibles (las originales se perdieron en el one-hot) para describir los grupos
d["producto"] = df_clean["producto"].values; d["canal"] = df_clean["canal"].values
R["dist_objetivo"] = d.entrega_rapida.value_counts(normalize=True).round(3).to_dict()
dm = pd.get_dummies(d.drop(columns=["producto","canal"]), columns=["ciudad"], prefix="ciudad")
feats = ["cantidad","mes_pedido","pedido_grande"] + [c for c in dm.columns if c.startswith(("canal_","producto_","ciudad_"))]
X, y = dm[feats], dm["entrega_rapida"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=SEED, stratify=y)
R["n_train"], R["n_test"], R["n_features"] = len(Xtr), len(Xte), len(feats)
lr = LogisticRegression(max_iter=1000, random_state=SEED).fit(Xtr, ytr)
yp = lr.predict(Xte)
R["lr"] = {"accuracy": accuracy_score(yte, yp), "precision": precision_score(yte, yp, zero_division=0),
           "recall": recall_score(yte, yp, zero_division=0), "f1": f1_score(yte, yp, zero_division=0)}
base = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
R["baseline_accuracy"] = accuracy_score(yte, base.predict(Xte))
R["cm"] = confusion_matrix(yte, yp).tolist()
coef = pd.DataFrame({"variable": feats, "coeficiente": lr.coef_[0]}).sort_values("coeficiente", ascending=False)
coef.to_csv("tablas/coeficientes_logistica.csv", index=False)
fig, ax = plt.subplots(figsize=(5, 4.3))
sns.heatmap(confusion_matrix(yte, yp), annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Pred: No rápida","Pred: Rápida"], yticklabels=["Real: No rápida","Real: Rápida"])
ax.set_title("Matriz de confusión – entrega rápida"); plt.tight_layout(); plt.savefig("graficas/matriz_confusion.png", dpi=130); plt.close()
fig, ax = plt.subplots(figsize=(6.5, 4.2))
c = coef.set_index("variable").coeficiente.sort_values()
c.plot(kind="barh", ax=ax, color=np.where(c < 0, "#C0504D", "#4472C4")); ax.set_title("Coeficientes de la regresión logística"); ax.set_ylabel("")
plt.tight_layout(); plt.savefig("graficas/coeficientes.png", dpi=130); plt.close()
# Complemento: validación cruzada estratificada (más estable que una sola partición de 51 pedidos)
from sklearn.model_selection import StratifiedKFold, cross_val_score
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
R["cv_lr_acc"] = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=cv, scoring="accuracy").tolist()
R["cv_base_acc"] = cross_val_score(DummyClassifier(strategy="most_frequent"), X, y, cv=cv, scoring="accuracy").tolist()

# ============================ AGRUPAMIENTO NO SUPERVISADO ============================
vars_k = ["cantidad","precio_unitario","tiempo_entrega_dias","calificacion"]
may = d[d.cantidad >= 25]; R["mayoristas"] = may.cantidad.tolist()
dk = d[d.cantidad < 25].copy().reset_index(drop=True)
R["n_cluster"] = len(dk)
R["desc_vars"] = dk[vars_k].describe().round(2).to_dict()
Z = StandardScaler().fit_transform(dk[vars_k])
filas = []
for k in range(2, 8):
    km = KMeans(k, n_init=10, random_state=SEED).fit(Z); lab = km.labels_
    filas.append({"k": k, "inercia": km.inertia_, "silueta": silhouette_score(Z, lab), "grupo_min": int(np.bincount(lab).min())})
sel = pd.DataFrame(filas); sel.to_csv("tablas/seleccion_k.csv", index=False); R["seleccion_k"] = sel.round(3).to_dict("records")
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot(sel.k, sel.inercia, "o-"); ax[0].set_title("Método del codo (inercia)"); ax[0].set_xlabel("k")
ax[1].plot(sel.k, sel.silueta, "o-", color="#ED7D31"); ax[1].set_title("Coeficiente de silueta"); ax[1].set_xlabel("k")
ax[2].bar(sel.k, sel.grupo_min, color="#70AD47"); ax[2].set_title("Tamaño del grupo más pequeño"); ax[2].set_xlabel("k")
plt.tight_layout(); plt.savefig("graficas/codo_silueta.png", dpi=130); plt.close()

km = KMeans(4, n_init=10, random_state=SEED).fit(Z); dk["cluster"] = km.labels_
R["silueta_k4"] = float(silhouette_score(Z, km.labels_)); R["tam_clusters"] = np.bincount(km.labels_).tolist()
perfil = dk.groupby("cluster")[vars_k].mean().round(2); perfil["pedidos"] = dk.cluster.value_counts().sort_index()
perfil["pct_ingresos"] = (dk.groupby("cluster").apply(lambda g: (g.cantidad*g.precio_unitario).sum()) / (dk.cantidad*dk.precio_unitario).sum() * 100).round(1)
perfil["pct_entrega_rapida"] = (dk.groupby("cluster").entrega_rapida.mean()*100).round(1)
# Nombres según el perfil (no según el número del algoritmo)
nombres = {}
nombres[perfil.tiempo_entrega_dias.idxmax()] = "Entrega lenta"
rest = perfil.drop(index=list(nombres)); nombres[rest.cantidad.idxmax()] = "Multiunidad"
rest = perfil.drop(index=list(nombres)); nombres[rest.precio_unitario.idxmax()] = "Premium"
rest = perfil.drop(index=list(nombres)); nombres[rest.index[0]] = "Estándar"
dk["grupo"] = dk.cluster.map(nombres); perfil["grupo"] = perfil.index.map(nombres)
perfil = perfil.set_index("grupo").loc[["Estándar","Premium","Multiunidad","Entrega lenta"]]
perfil.to_csv("tablas/perfil_grupos.csv"); R["perfil"] = perfil.reset_index().to_dict("records")
dk.to_csv("data/ARGCOL_pedidos_con_grupo.csv", index=False, encoding="utf-8")

orden = ["Estándar","Premium","Multiunidad","Entrega lenta"]
zper = pd.DataFrame(Z, columns=vars_k); zper["grupo"] = dk.grupo
hm = zper.groupby("grupo")[vars_k].mean().loc[orden]
fig, ax = plt.subplots(figsize=(6.5, 3.6)); sns.heatmap(hm, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Perfil estandarizado de cada grupo"); ax.set_ylabel(""); plt.tight_layout(); plt.savefig("graficas/mapa_calor_grupos.png", dpi=130); plt.close()

pca = PCA(2, random_state=SEED); P = pca.fit_transform(Z); R["pca_var"] = pca.explained_variance_ratio_.round(3).tolist()
R["pca_cargas"] = pd.DataFrame(pca.components_.T, index=vars_k, columns=["PC1","PC2"]).round(2).to_dict()
fig, ax = plt.subplots(figsize=(6.5, 4.6))
for g, col in zip(orden, ["#4472C4","#ED7D31","#70AD47","#C0504D"]):
    m = (dk.grupo == g).values; ax.scatter(P[m,0], P[m,1], s=22, alpha=.75, label=g, color=col)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f} %)"); ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f} %)")
ax.set_title("Pedidos proyectados con PCA"); ax.legend(); plt.tight_layout(); plt.savefig("graficas/pca_grupos.png", dpi=130); plt.close()

comp = pd.crosstab(dk.grupo, dk.producto, normalize="index").loc[orden] * 100
fig, ax = plt.subplots(figsize=(7, 3.8)); comp.plot(kind="bar", stacked=True, ax=ax, colormap="tab20c")
ax.set_ylabel("% de pedidos"); ax.set_xlabel(""); ax.set_title("Composición de cada grupo por producto"); ax.legend(fontsize=7, bbox_to_anchor=(1, 1))
plt.xticks(rotation=0); plt.tight_layout(); plt.savefig("graficas/composicion_producto.png", dpi=130); plt.close()
R["composicion_producto"] = comp.round(1).to_dict("index")

# Verificación: entrega lenta por canal/ciudad/mes + chi-cuadrado
dk["lenta"] = (dk.grupo == "Entrega lenta").astype(int); R["pct_lenta_global"] = round(dk.lenta.mean()*100, 1)
chi = {}
for v in ["canal","ciudad","mes_pedido"]:
    ct = pd.crosstab(dk[v], dk.lenta); chi[v] = {"p": round(float(chi2_contingency(ct)[1]), 3),
        "pct_lenta": (dk.groupby(v).lenta.mean()*100).round(1).to_dict()}
R["chi2"] = chi
# Estabilidad: ARI con distintas semillas + sensibilidad a mayoristas
base_lab = KMeans(4, n_init=10, random_state=SEED).fit_predict(Z)
R["ari"] = [round(adjusted_rand_score(base_lab, KMeans(4, n_init=10, random_state=s).fit_predict(Z)), 3) for s in (0, 1, 7, 123)]
Zall = StandardScaler().fit_transform(d[vars_k]); R["tam_con_mayoristas"] = np.bincount(KMeans(4, n_init=10, random_state=SEED).fit_predict(Zall)).tolist()
# Comparación MinMax vs Standard (complemento)
from sklearn.preprocessing import MinMaxScaler
R["ari_minmax"] = round(adjusted_rand_score(base_lab, KMeans(4, n_init=10, random_state=SEED).fit_predict(MinMaxScaler().fit_transform(dk[vars_k]))), 3)

json.dump(R, open("tablas/resultados.json", "w"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
