# 🔮 ChurnPredictor Retail — Predicción de Abandono de Clientes

Modelo de Machine Learning para predecir qué clientes de una cadena retail
van a abandonar antes de que ocurra, con segmentación de riesgo y
recomendaciones de negocio accionables.

**Conecta directamente con el proyecto RetailBI** de este portafolio —
usa el mismo dominio de negocio y estructura de datos.

---

## 🎯 Problema de Negocio

Una cadena retail pierde en promedio el **27–32% de sus clientes** cada año.
Retener un cliente existente cuesta 5–7× menos que adquirir uno nuevo.

**Objetivo:** Predecir con al menos 85% de AUC-ROC qué clientes tienen alta
probabilidad de no volver a comprar en los próximos 90 días, para activar
campañas de retención a tiempo.

**Definición de Churn:** Cliente que lleva más de 90 días sin realizar ninguna compra.

---

## 📊 Resultados del Modelo

| Modelo | AUC-ROC | F1 Score | Precision | Recall |
|--------|---------|----------|-----------|--------|
| **Random Forest** ⭐ | **0.857** | **0.817** | 0.897 | 0.750 |
| Logistic Regression | 0.855 | 0.809 | 0.888 | 0.742 |
| XGBoost | 0.848 | 0.803 | 0.887 | 0.734 |

*Evaluados con StratifiedKFold 5-fold + test set 20% no visto.*

### Segmentación de riesgo (2,000 clientes)

| Segmento | Clientes | Churn real | Acción recomendada |
|----------|------:|-------:|-------------------|
| Riesgo bajo | 1,147 | 2.9% | Mantener programa de fidelización |
| Riesgo medio | 300 | 42.3% | Campaña de nurturing |
| Riesgo alto | 38 | 57.9% | Descuento personalizado |
| **Riesgo crítico** | **515** | **88.7%** | **Contacto inmediato** |

**Revenue en riesgo estimado: COP 1,160,528,707**
**Revenue recuperable (retención 30%): COP ~348M**

---

## 🧠 Features de Mayor Impacto

1. `recencia_dias` — días desde la última compra *(correlación 0.88 con churn)*
2. `compras_ultimo_trim` — compras en últimos 90 días *(correlación -0.73)*
3. `duracion_relacion_dias` — antigüedad del cliente *(correlación -0.68)*
4. `nps_score` — satisfacción del cliente *(correlación -0.50)*
5. `max_gap_entre_compras` — mayor período sin comprar

**Señales de alerta temprana:**
- Recencia > 60 días
- Compras último trimestre = 0
- NPS Score ≤ 5
- Gap máximo entre compras > 90 días

---


## ⚙️ Pipeline de Machine Learning

```
Datos          →  Feature          →  Preprocesamiento   →  Modelos
retail            Engineering         ColumnTransformer      LR / RF / XGB
                  RFM + comportam.    StandardScaler         CV 5-fold
                  13 features num.    OneHotEncoder          Threshold óptimo
                  2 features cat.     StratifiedSplit        por F1
                                                              ↓
                                                          Segmentación
                                                          de riesgo 4 niveles
                                                              ↓
                                                          Recomendaciones
                                                          de negocio
```

---


## 🔗 Conexión con el Portafolio

Este proyecto es parte de una serie de 4 proyectos conectados:

| # | Proyecto | Conexión |
|---|----------|----------|
| 1 | EDA Transporte Urbano | Análisis exploratorio base |
| 2 | RetailBI — Power BI | Mismo dominio retail, datos de ventas |
| 3 | ClinicaAPI — FastAPI | Capa de servicio de datos |
| 4 | **ChurnPredictor** ← | Usa la misma estructura de clientes del RetailBI |

---

## 👤 Autor

**Ricardo Javier Sequeda Goez**
Data Analyst | ML | Python & SQL
📧 Ricardojgoez@gmail.com | [LinkedIn](https://linkedin.com/in/ricardosequeda)
