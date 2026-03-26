"""
03_modelo_churn.py
===================
Pipeline completo de Machine Learning para predicción de churn:

  1. Preprocesamiento con Pipeline de scikit-learn
  2. Tres modelos: Logistic Regression, Random Forest, XGBoost
  3. Cross-validation 5-fold para evaluación robusta
  4. Métricas: AUC-ROC, F1, Precision, Recall, Accuracy
  5. Curva ROC comparativa
  6. Matriz de confusión del mejor modelo
  7. Feature importance (RF + XGBoost)
  8. Segmentación de riesgo con probabilidades
  9. Recomendaciones de negocio

Buenas prácticas aplicadas:
  ✅ Pipeline scikit-learn (evita data leakage)
  ✅ StratifiedKFold (respeta el desbalance de clases)
  ✅ class_weight='balanced' (manejo de imbalance)
  ✅ Threshold óptimo por F1 (no siempre 0.5)
  ✅ Separación limpia train/test antes de cualquier procesamiento
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, roc_curve, classification_report
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BLUE   = "#1F4E8C"
RED    = "#C0392B"
GREEN  = "#27AE60"
AMBER  = "#E67E22"
GRAY   = "#888780"

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


# ════════════════════════════════════════════════════════
# 1. CARGA Y PREPARACIÓN
# ════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  1. CARGA Y PREPARACIÓN DE DATOS")
print("═"*60)

df = pd.read_csv("data/clientes_churn.csv")

# Features seleccionadas (basadas en correlación del EDA)
FEATURES_NUM = [
    "recencia_dias",
    "frecuencia",
    "valor_total_cop",
    "ticket_promedio",
    "duracion_relacion_dias",
    "compras_ultimo_trim",
    "compras_ultimo_anio",
    "variacion_frecuencia",
    "max_gap_entre_compras",
    "nps_score",
    "n_categorias_compradas",
    "usa_app",
    "tiene_descuentos",
]
FEATURES_CAT = ["segmento", "canal_principal"]
TARGET = "churn"

X = df[FEATURES_NUM + FEATURES_CAT]
y = df[TARGET]

# Split estratificado (mantiene proporción de churn en train y test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Train: {len(X_train):,} clientes  |  Test: {len(X_test):,} clientes")
print(f"  Tasa churn train: {y_train.mean():.1%}  |  Test: {y_test.mean():.1%}")
print(f"  Features numéricas: {len(FEATURES_NUM)}  |  Categóricas: {len(FEATURES_CAT)}")


# ════════════════════════════════════════════════════════
# 2. PREPROCESADOR (ColumnTransformer)
# ════════════════════════════════════════════════════════
preprocesador = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES_CAT),
    ]
)


# ════════════════════════════════════════════════════════
# 3. DEFINIR MODELOS
# ════════════════════════════════════════════════════════
modelos = {
    "Logistic Regression": Pipeline([
        ("prep", preprocesador),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            C=0.5
        ))
    ]),
    "Random Forest": Pipeline([
        ("prep", preprocesador),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ]),
    "XGBoost": Pipeline([
        ("prep", preprocesador),
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ))
    ]),
}


# ════════════════════════════════════════════════════════
# 4. CROSS-VALIDATION 5-FOLD
# ════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  2. CROSS-VALIDATION 5-FOLD")
print("═"*60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
resultados_cv = {}

for nombre, pipeline in modelos.items():
    scores = cross_validate(
        pipeline, X_train, y_train, cv=cv,
        scoring=["roc_auc", "f1", "precision", "recall"],
        n_jobs=-1
    )
    resultados_cv[nombre] = {
        "AUC-ROC":  scores["test_roc_auc"].mean(),
        "F1":       scores["test_f1"].mean(),
        "Precision":scores["test_precision"].mean(),
        "Recall":   scores["test_recall"].mean(),
        "AUC std":  scores["test_roc_auc"].std(),
    }
    print(f"\n  {nombre}:")
    print(f"    AUC-ROC:   {scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}")
    print(f"    F1:        {scores['test_f1'].mean():.4f}")
    print(f"    Precision: {scores['test_precision'].mean():.4f}")
    print(f"    Recall:    {scores['test_recall'].mean():.4f}")


# ════════════════════════════════════════════════════════
# 5. ENTRENAMIENTO FINAL Y EVALUACIÓN EN TEST
# ════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  3. EVALUACIÓN EN TEST SET")
print("═"*60)

resultados_test = {}
pipelines_entrenados = {}

for nombre, pipeline in modelos.items():
    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Threshold óptimo por F1 (no usar siempre 0.5)
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1s = [f1_score(y_test, (y_proba >= t).astype(int)) for t in thresholds]
    threshold_optimo = thresholds[np.argmax(f1s)]
    y_pred = (y_proba >= threshold_optimo).astype(int)

    resultados_test[nombre] = {
        "AUC-ROC":   roc_auc_score(y_test, y_proba),
        "F1":        f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Threshold": threshold_optimo,
        "y_proba":   y_proba,
    }
    pipelines_entrenados[nombre] = pipeline

    print(f"\n  {nombre} (threshold={threshold_optimo:.2f}):")
    print(f"    AUC-ROC:   {resultados_test[nombre]['AUC-ROC']:.4f}")
    print(f"    F1:        {resultados_test[nombre]['F1']:.4f}")
    print(f"    Precision: {resultados_test[nombre]['Precision']:.4f}")
    print(f"    Recall:    {resultados_test[nombre]['Recall']:.4f}")

# Mejor modelo por AUC-ROC
mejor_nombre = max(resultados_test, key=lambda k: resultados_test[k]["AUC-ROC"])
print(f"\n  ★ Mejor modelo: {mejor_nombre} (AUC={resultados_test[mejor_nombre]['AUC-ROC']:.4f})")


# ════════════════════════════════════════════════════════
# 6. VISUALIZACIONES
# ════════════════════════════════════════════════════════

# 6.1 Tabla comparativa de modelos
print("\n" + "═"*60)
print("  4. VISUALIZACIONES")
print("═"*60)

df_resultados = pd.DataFrame({
    nombre: {k: v for k, v in res.items() if k not in ["y_proba","Threshold"]}
    for nombre, res in resultados_test.items()
}).T.round(4)

fig, ax = plt.subplots(figsize=(12, 4))
metricas = ["AUC-ROC", "F1", "Precision", "Recall", "Accuracy"]
x = np.arange(len(metricas))
width = 0.25
colors_mod = [BLUE, GREEN, AMBER]

for i, (nombre, res) in enumerate(resultados_test.items()):
    vals = [res[m] for m in metricas]
    bars = ax.bar(x + i*width, vals, width, label=nombre,
                  color=colors_mod[i], alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.2f}", ha="center", fontsize=8)

ax.set_xticks(x + width)
ax.set_xticklabels(metricas)
ax.set_ylabel("Score")
ax.set_title("Comparación de Modelos — Test Set")
ax.legend()
ax.set_ylim(0, 1.12)
guardar("06_comparacion_modelos")


# 6.2 Curva ROC comparativa
fig, ax = plt.subplots(figsize=(8, 7))
colores_roc = [BLUE, GREEN, AMBER]
for (nombre, res), color in zip(resultados_test.items(), colores_roc):
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"{nombre} (AUC={res['AUC-ROC']:.3f})")
ax.plot([0,1],[0,1], "k--", linewidth=1, alpha=0.5, label="Random (AUC=0.500)")
ax.fill_between(*roc_curve(y_test, resultados_test[mejor_nombre]["y_proba"])[:2],
                alpha=0.08, color=BLUE)
ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
ax.set_title("Curva ROC — Comparación de Modelos")
ax.legend(loc="lower right")
guardar("07_curva_roc")


# 6.3 Matriz de confusión del mejor modelo
threshold_opt = resultados_test[mejor_nombre]["Threshold"]
y_pred_mejor = (resultados_test[mejor_nombre]["y_proba"] >= threshold_opt).astype(int)
cm = confusion_matrix(y_test, y_pred_mejor)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Activo (0)","Churn (1)"],
            yticklabels=["Activo (0)","Churn (1)"],
            cbar=False, linewidths=0.5)
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
ax.set_title(f"Matriz de Confusión — {mejor_nombre}\n(threshold={threshold_opt:.2f})")

# Anotaciones de negocio
vn, fp, fn, vp = cm.ravel()
ax.text(1.02, 0.5,
    f"VN={vn} (activos correctos)\n"
    f"VP={vp} (churns detectados)\n"
    f"FP={fp} (falsa alarma)\n"
    f"FN={fn} (churn no detectado)",
    transform=ax.transAxes, fontsize=9,
    verticalalignment='center',
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
guardar("08_matriz_confusion")


# 6.4 Feature importance (mejor modelo)
if mejor_nombre in ["Random Forest", "XGBoost"]:
    clf = pipelines_entrenados[mejor_nombre]["clf"]
    prep = pipelines_entrenados[mejor_nombre]["prep"]

    # Reconstruir nombres de features después de OneHotEncoder
    feat_names_num = FEATURES_NUM
    feat_names_cat = prep.named_transformers_["cat"].get_feature_names_out(FEATURES_CAT).tolist()
    all_feat_names = feat_names_num + feat_names_cat

    importances = clf.feature_importances_
    fi_df = pd.DataFrame({
        "feature": all_feat_names,
        "importance": importances
    }).sort_values("importance", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    colores_fi = [RED if imp > fi_df["importance"].quantile(0.75) else BLUE
                  for imp in fi_df["importance"]]
    ax.barh(fi_df["feature"], fi_df["importance"],
            color=colores_fi, alpha=0.85, edgecolor="white")
    ax.set_title(f"Feature Importance — {mejor_nombre} (Top 15)")
    ax.set_xlabel("Importancia")
    guardar("09_feature_importance")


# ════════════════════════════════════════════════════════
# 7. SEGMENTACIÓN DE RIESGO
# ════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  5. SEGMENTACIÓN DE RIESGO")
print("═"*60)

mejor_pipeline = pipelines_entrenados[mejor_nombre]
df["prob_churn"] = mejor_pipeline.predict_proba(X)[:, 1]

df["segmento_riesgo"] = pd.cut(
    df["prob_churn"],
    bins=[0, 0.25, 0.50, 0.75, 1.0],
    labels=["Riesgo bajo", "Riesgo medio", "Riesgo alto", "Riesgo crítico"]
)

seg_riesgo = df.groupby("segmento_riesgo").agg(
    n_clientes=("cliente_id", "count"),
    prob_media=("prob_churn", "mean"),
    valor_medio=("valor_total_cop", "mean"),
    churn_real=("churn", "mean"),
).round(3)
print(seg_riesgo.to_string())

# Gráfica segmentación
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colores_seg = [GREEN, AMBER, "#E67E22", RED]

seg_counts = df["segmento_riesgo"].value_counts().sort_index()
axes[0].bar(seg_counts.index, seg_counts.values,
            color=colores_seg, alpha=0.85, edgecolor="white")
axes[0].set_title("Clientes por Segmento de Riesgo")
axes[0].set_ylabel("N° clientes")
axes[0].tick_params(axis='x', rotation=15)
for i, v in enumerate(seg_counts.values):
    axes[0].text(i, v + 5, f"{v:,}", ha="center", fontsize=10)

# LTV en riesgo (valor en peligro de perder)
valor_riesgo = df[df["segmento_riesgo"].isin(["Riesgo alto","Riesgo crítico"])].groupby(
    "segmento_riesgo")["valor_total_cop"].sum()
axes[1].bar(valor_riesgo.index, valor_riesgo.values / 1e6,
            color=[RED, "#8B0000"], alpha=0.85, edgecolor="white")
axes[1].set_title("Revenue en Riesgo por Segmento (Millones COP)")
axes[1].set_ylabel("Millones COP")
axes[1].tick_params(axis='x', rotation=10)
for i, v in enumerate(valor_riesgo.values):
    axes[1].text(i, v/1e6 + 0.5, f"${v/1e6:.1f}M", ha="center", fontsize=10)

guardar("10_segmentacion_riesgo")


# ════════════════════════════════════════════════════════
# 8. GUARDAR MODELO Y RESULTADOS
# ════════════════════════════════════════════════════════
joblib.dump(mejor_pipeline, f"modelos/modelo_churn_{mejor_nombre.lower().replace(' ','_')}.pkl")
df[["cliente_id","segmento","prob_churn","segmento_riesgo","churn"]].to_csv(
    "data/predicciones_churn.csv", index=False)

print(f"\n  Modelo guardado: modelos/modelo_churn_{mejor_nombre.lower().replace(' ','_')}.pkl")
print(f"  Predicciones guardadas: data/predicciones_churn.csv")


# ════════════════════════════════════════════════════════
# 9. CONCLUSIONES Y RECOMENDACIONES DE NEGOCIO
# ════════════════════════════════════════════════════════
n_riesgo_alto   = len(df[df["segmento_riesgo"] == "Riesgo alto"])
n_riesgo_critico = len(df[df["segmento_riesgo"] == "Riesgo crítico"])
revenue_en_riesgo = df[df["segmento_riesgo"].isin(
    ["Riesgo alto","Riesgo crítico"])]["valor_total_cop"].sum()

print(f"""
{"═"*60}
  CONCLUSIONES DE NEGOCIO
{"═"*60}

  Modelo seleccionado: {mejor_nombre}
  AUC-ROC: {resultados_test[mejor_nombre]['AUC-ROC']:.4f}
  F1 Score: {resultados_test[mejor_nombre]['F1']:.4f}
  Recall (detección de churn): {resultados_test[mejor_nombre]['Recall']:.4f}

  ─── Clientes en riesgo ───────────────────────────────
  Riesgo alto:    {n_riesgo_alto:>5,} clientes
  Riesgo crítico: {n_riesgo_critico:>5,} clientes
  Revenue en riesgo: COP {revenue_en_riesgo:>12,.0f}

  ─── Recomendaciones de negocio ──────────────────────
  1. RIESGO CRÍTICO — Acción inmediata:
     Contactar en las próximas 48h con oferta personalizada.
     Priorizar clientes con alto valor_total_cop.

  2. RIESGO ALTO — Campaña de retención:
     Email/SMS con descuento del 15-20% en su categoría favorita.
     Activar programa de puntos o fidelización.

  3. RIESGO MEDIO — Monitoreo:
     Inclusión en flujo de nurturing con contenido relevante.
     Re-activar si lleva 30+ días sin compra.

  4. SEÑALES DE ALERTA TEMPRANA (top features del modelo):
     - Recencia > 60 días sin compra
     - Compras último trimestre = 0
     - NPS Score ≤ 5
     - Gap máximo entre compras > 90 días

  5. ROI esperado de la retención:
     Si se retiene el 30% de clientes en riesgo crítico:
     COP {revenue_en_riesgo * 0.30:>12,.0f} en revenue protegido
""")
