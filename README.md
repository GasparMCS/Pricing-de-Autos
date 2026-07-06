# Pricing de Autos 🚗

Proyecto de Ciencia de Datos (UDP) — predicción de precios de autos usados en Chile y
una **interfaz web interactiva** para la feria de proyectos.

El usuario ingresa las características de su auto (marca, modelo, año, kilometraje,
combustible, transmisión) y el precio al que lo quiere vender o encontró publicado; la
app predice el precio de mercado con el modelo XGBoost entrenado y dice si está
**subvalorado, en precio justo o sobrevalorado**. Además muestra un gráfico de
contribuciones (SHAP waterfall) que explica *por qué* el modelo predice ese precio.

## Estructura

```
interfaz-feria/     Interfaz web (corre 100% en el navegador, sin servidor)
  index.html          UI
  app.js              Motor de inferencia (replica el XGBoost en JS)
  modelo_web.js       Modelo real exportado (árboles + encoders) en JSON
  exportar_modelo.py  Celda de Colab que genera modelo_web.js desde el notebook
notebooks/
  Entrega_2.ipynb     Notebook final (EDA, limpieza, modelado, evaluación)
scripts/            Scripts auxiliares (scraping, modelos, VIF)
data/               Datasets del modelo (limpios + scraped)
```

## Cómo abrir la interfaz

Basta con abrir `interfaz-feria/index.html` en un navegador (doble clic).
No necesita servidor ni conexión: el modelo va embebido en `modelo_web.js`.

## Modelo

- **Algoritmo:** XGBoost Regressor (300 árboles, `max_depth=7`, `learning_rate=0.1`).
- **Target:** `log1p(price)` (se revierte con `expm1`).
- **Features:** antigüedad, kilometraje, marca y modelo (target encoding),
  combustible y transmisión (one-hot).
- **Desempeño (test):** R² ≈ 0.816 · MAPE ≈ 18.6 %.

La inferencia en el navegador reproduce el modelo **bit a bit** (comparaciones en
`float32` con `Math.fround`), por lo que la predicción web coincide exactamente con
`model.predict` del notebook.

---
Grupo 3 — Ciencia de Datos, Universidad Diego Portales.
