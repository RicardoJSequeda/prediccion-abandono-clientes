"""
01_generar_datos.py
====================
Genera un dataset de comportamiento de clientes retail para predecir churn.
Diseñado para conectar con el RetailBI existente en el portafolio.

Ejecutar: python 01_generar_datos.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random, os

np.random.seed(42)
random.seed(42)

os.makedirs("data", exist_ok=True)
os.makedirs("graficas", exist_ok=True)
os.makedirs("modelos", exist_ok=True)

FECHA_HOY   = datetime(2025, 3, 1)
FECHA_INICIO = datetime(2023, 1, 1)
N_CLIENTES  = 2000

SEGMENTOS   = ["Premium","Regular","Ocasional","Corporativo","Online"]
CANALES     = ["Tienda física","E-commerce","App móvil","Teléfono"]
CIUDADES    = ["Bogotá","Medellín","Cali","Barranquilla","Montería",
               "Cartagena","Bucaramanga","Pereira","Cúcuta","Ibagué"]
CATEGORIAS  = ["Tecnología","Ropa","Calzado","Hogar","Deportes","Belleza","Libros"]

def generar_cliente(cid, segmento):
    """Genera historial de compras con patrones realistas por segmento."""

    # Parámetros por segmento
    params = {
        "Premium":     {"freq_base": 8,  "ticket_base": 350_000, "churn_prob": 0.12},
        "Regular":     {"freq_base": 5,  "ticket_base": 180_000, "churn_prob": 0.25},
        "Ocasional":   {"freq_base": 2,  "ticket_base": 90_000,  "churn_prob": 0.45},
        "Corporativo": {"freq_base": 12, "ticket_base": 800_000, "churn_prob": 0.10},
        "Online":      {"freq_base": 6,  "ticket_base": 150_000, "churn_prob": 0.30},
    }
    p = params[segmento]

    # ¿Este cliente hace churn?
    es_churn = np.random.random() < p["churn_prob"]

    # Número total de compras en el período
    n_compras = max(1, int(np.random.poisson(p["freq_base"] * 2)))

    # Si hace churn, sus compras se concentran en la primera mitad del período
    dias_total = (FECHA_HOY - FECHA_INICIO).days
    if es_churn:
        # Última compra entre 90 y 400 días atrás
        ultimo_dias = random.randint(90, 400)
        # Compras concentradas en período activo
        fechas_compras = sorted([
            FECHA_HOY - timedelta(days=random.randint(ultimo_dias, dias_total))
            for _ in range(n_compras)
        ])
    else:
        # Compras recientes, con última compra en los últimos 89 días
        ultimo_dias = random.randint(1, 89)
        fechas_compras = sorted([
            FECHA_HOY - timedelta(days=random.randint(ultimo_dias, dias_total))
            for _ in range(n_compras)
        ])
        # Garantizar al menos una compra reciente
        fechas_compras[-1] = FECHA_HOY - timedelta(days=ultimo_dias)

    # Generar montos con variabilidad
    montos = [
        max(15_000, int(np.random.normal(p["ticket_base"], p["ticket_base"] * 0.4)))
        for _ in fechas_compras
    ]

    ultima_compra = max(fechas_compras)
    primera_compra = min(fechas_compras)
    dias_inactivo = (FECHA_HOY - ultima_compra).days
    duracion_relacion = (ultima_compra - primera_compra).days + 1

    return {
        "cliente_id":       f"CLI{cid:05d}",
        "segmento":         segmento,
        "canal_principal":  random.choice(CANALES),
        "ciudad":           random.choice(CIUDADES),
        "categoria_favorita": random.choice(CATEGORIAS),
        "fecha_primera_compra": primera_compra.date(),
        "fecha_ultima_compra":  ultima_compra.date(),

        # ── Features RFM ──────────────────────────────────────
        "recencia_dias":    dias_inactivo,                          # R
        "frecuencia":       n_compras,                              # F
        "valor_total_cop":  sum(montos),                            # M
        "ticket_promedio":  int(np.mean(montos)),

        # ── Features de comportamiento ──────────────────────
        "duracion_relacion_dias": duracion_relacion,
        "compras_ultimo_trim":  sum(
            1 for f in fechas_compras if (FECHA_HOY - f).days <= 90
        ),
        "compras_ultimo_anio":  sum(
            1 for f in fechas_compras if (FECHA_HOY - f).days <= 365
        ),
        "variacion_frecuencia": round(
            (n_compras / max(1, duracion_relacion / 30)), 4
        ),  # compras por mes
        "max_gap_entre_compras": max(
            [(fechas_compras[i+1] - fechas_compras[i]).days
             for i in range(len(fechas_compras)-1)], default=dias_inactivo
        ),
        "tiene_descuentos":  random.choices([1, 0], weights=[0.6, 0.4])[0],
        "n_categorias_compradas": random.randint(1, min(4, n_compras)),
        "usa_app":           1 if random.random() < 0.45 else 0,
        "nps_score":         random.choices(
            [1,2,3,4,5,6,7,8,9,10],
            weights=[2,2,3,4,6,8,10,18,22,25]
        )[0] if not es_churn else random.choices(
            [1,2,3,4,5,6,7,8,9,10],
            weights=[10,12,15,14,12,10,10,8,6,3]
        )[0],

        # ── Variable objetivo ─────────────────────────────────
        "churn": int(es_churn),
    }


print("Generando dataset de clientes...")
pesos_segmentos = [0.12, 0.42, 0.22, 0.10, 0.14]
segmentos_asignados = np.random.choice(SEGMENTOS, size=N_CLIENTES, p=pesos_segmentos)

clientes = [generar_cliente(i+1, seg) for i, seg in enumerate(segmentos_asignados)]
df = pd.DataFrame(clientes)

df.to_csv("data/clientes_churn.csv", index=False)

# Resumen
tasa_churn = df["churn"].mean()
print(f"  Dataset: {len(df):,} clientes")
print(f"  Tasa de churn: {tasa_churn:.1%}")
print(f"  Churn por segmento:")
print(df.groupby("segmento")["churn"].agg(["sum","mean"]).rename(
    columns={"sum":"cantidad","mean":"tasa"}).assign(
    tasa=lambda x: x["tasa"].map("{:.1%}".format)).to_string())
print(f"\n  Archivo guardado: data/clientes_churn.csv")
