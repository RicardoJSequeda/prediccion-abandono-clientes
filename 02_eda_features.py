"""
02_eda_features.py
===================
Análisis exploratorio enfocado en churn:
  - Distribución de la variable objetivo
  - Análisis RFM por segmento de riesgo
  - Correlaciones con churn
  - Visualización de separabilidad de features

Ejecutar: python 02_eda_features.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

BLUE   = "#1F4E8C"
RED    = "#C0392B"
GREEN  = "#27AE60"
AMBER  = "#E67E22"
GRAY   = "#888780"
PALETTE = [BLUE, RED]

sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({
    "figure.dpi": 130, "font.family": "DejaVu Sans",
    "axes.titlesize": 13, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

def guardar(nombre):
    plt.tight_layout()
    plt.savefig(f"graficas/{nombre}.png", bbox_inches="tight")
    plt.close()
    print(f"  ✅ graficas/{nombre}.png")


df = pd.read_csv("data/clientes_churn.csv")

print("═"*60)
print("  EDA — ANÁLISIS DE CHURN")
print("═"*60)
print(f"\n  Clientes totales:  {len(df):,}")
print(f"  Tasa de churn:     {df['churn'].mean():.1%}")
print(f"  Features:          {df.shape[1]-1}")


# ── 1. Distribución de churn + por segmento ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Dona
counts  = df["churn"].value_counts()
colors  = [GREEN, RED]
wedges, texts, autotexts = axes[0].pie(
    counts, labels=["Activo","Churn"], colors=colors,
    autopct="%1.1f%%", startangle=90,
    wedgeprops={"width":0.5, "edgecolor":"white", "linewidth":2}
)
for at in autotexts: at.set_fontsize(12)
axes[0].set_title("Distribución de Churn")

# Por segmento
seg_churn = df.groupby("segmento")["churn"].mean().sort_values()
colores_seg = [RED if v > 0.30 else AMBER if v > 0.18 else GREEN for v in seg_churn.values]
bars = axes[1].barh(seg_churn.index, seg_churn.values * 100,
                    color=colores_seg, alpha=0.85, edgecolor="white")
for bar, v in zip(bars, seg_churn.values):
    axes[1].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{v:.1%}", va="center", fontsize=10)
axes[1].set_title("Tasa de Churn por Segmento")
axes[1].set_xlabel("% de clientes que hacen churn")
axes[1].axvline(df["churn"].mean() * 100, color=GRAY, linestyle="--",
                linewidth=1, label=f"Promedio: {df['churn'].mean():.1%}")
axes[1].legend(fontsize=9)
guardar("01_distribucion_churn")


# ── 2. Análisis RFM: Recencia, Frecuencia, Valor por estado ─────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
rfm_features = ["recencia_dias", "frecuencia", "ticket_promedio"]
rfm_labels   = ["Recencia (días inactivo)", "Frecuencia (# compras)", "Ticket Promedio (COP)"]
rfm_colors   = {0: GREEN, 1: RED}
rfm_labels_churn = {0: "Activo", 1: "Churn"}

for ax, feat, label in zip(axes, rfm_features, rfm_labels):
    for churn_val in [0, 1]:
        data = df[df["churn"] == churn_val][feat]
        ax.hist(data, bins=30, alpha=0.6, color=rfm_colors[churn_val],
                label=rfm_labels_churn[churn_val], density=True, edgecolor="white")
    ax.set_title(label)
    ax.set_xlabel(label)
    ax.set_ylabel("Densidad")
    ax.legend(fontsize=9)
    if feat == "ticket_promedio":
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

plt.suptitle("Distribución RFM: Activos vs Churn", fontsize=14, y=1.02)
guardar("02_rfm_distribucion")


# ── 3. Matriz de correlación con churn ──────────────────────────────────────
num_cols = [
    "recencia_dias","frecuencia","valor_total_cop","ticket_promedio",
    "duracion_relacion_dias","compras_ultimo_trim","compras_ultimo_anio",
    "variacion_frecuencia","max_gap_entre_compras","nps_score",
    "n_categorias_compradas","usa_app","tiene_descuentos","churn"
]
corr = df[num_cols].corr()["churn"].drop("churn").sort_values()

fig, ax = plt.subplots(figsize=(10, 6))
colores_corr = [RED if v > 0 else GREEN for v in corr.values]
bars = ax.barh(corr.index, corr.values, color=colores_corr, alpha=0.8, edgecolor="white")
for bar, v in zip(bars, corr.values):
    ax.text(v + (0.005 if v >= 0 else -0.005),
            bar.get_y() + bar.get_height()/2,
            f"{v:.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=9)
ax.axvline(0, color=GRAY, linewidth=0.8)
ax.set_title("Correlación de Variables con Churn", fontsize=13)
ax.set_xlabel("Correlación de Pearson")
guardar("03_correlacion_churn")


# ── 4. Boxplots de top features por churn ────────────────────────────────────
top_features = ["recencia_dias","compras_ultimo_trim","nps_score","max_gap_entre_compras"]
fig, axes = plt.subplots(1, 4, figsize=(16, 5))

for ax, feat in zip(axes, top_features):
    df.boxplot(column=feat, by="churn", ax=ax,
               patch_artist=True,
               boxprops=dict(alpha=0.7),
               medianprops=dict(color="white", linewidth=2))
    boxes = ax.patches
    for i, box in enumerate(boxes):
        box.set_facecolor(GREEN if i == 0 else RED)
    ax.set_title(feat.replace("_", " ").title())
    ax.set_xlabel("")
    ax.set_xticklabels(["Activo (0)", "Churn (1)"])

plt.suptitle("Top Features vs Estado Churn", fontsize=13, y=1.02)
guardar("04_boxplots_features")


# ── 5. Heatmap churn por ciudad y canal ──────────────────────────────────────
pivot = df.pivot_table(
    values="churn", index="ciudad",
    columns="canal_principal", aggfunc="mean"
).fillna(0)

fig, ax = plt.subplots(figsize=(11, 6))
sns.heatmap(pivot * 100, cmap="RdYlGn_r", ax=ax, annot=True,
            fmt=".0f", linewidths=0.5, cbar_kws={"label": "% churn"},
            vmin=0, vmax=60)
ax.set_title("Tasa de Churn (%) por Ciudad y Canal", fontsize=13, pad=15)
ax.set_xlabel("Canal principal")
ax.set_ylabel("")
guardar("05_churn_ciudad_canal")

print(f"\n  Top 3 features más correlacionadas con churn:")
print(f"  {corr.tail(3).to_string()}")
print(f"\n  Top 3 protectores contra churn:")
print(f"  {corr.head(3).to_string()}")
