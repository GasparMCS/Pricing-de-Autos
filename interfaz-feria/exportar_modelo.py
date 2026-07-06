# ══════════════════════════════════════════════════════════════════════════
# EXPORTAR MODELO PARA LA INTERFAZ WEB (feria de proyectos)
# Ejecutar DESPUÉS de correr todo el notebook Entrega 2.
# (necesita: xgb_best, X_train_enc, X_test_enc, y_train, y_test,
#            te_maps, te_globals, df_model, UMBRAL_INF, UMBRAL_SUP)
#
# NOTA: XGBoost trató las columnas One-Hot booleanas como CATEGÓRICAS, y su
# volcado JSON no incluye la regla de enrutamiento. Por eso reentrenamos el
# modelo ganador con esas columnas como enteros 0/1 (mismos hiperparámetros):
# quedan como splits numéricos exportables, con predicciones ~idénticas.
# ══════════════════════════════════════════════════════════════════════════
import json
import numpy as np
import xgboost as xgb

# 1. Castear columnas booleanas -> int (0/1) para que sean splits numéricos
def a_numerico(df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
    return df

Xtr = a_numerico(X_train_enc)
Xte = a_numerico(X_test_enc)

# 2. Reentrenar el modelo ganador con los MISMOS hiperparámetros
params = xgb_best.get_params()
params['enable_categorical'] = False
modelo_web = xgb.XGBRegressor(**params)
modelo_web.fit(Xtr, y_train)

booster = modelo_web.get_booster()
# with_stats=True agrega 'cover' a cada nodo -> necesario para el gráfico de
# contribuciones (SHAP waterfall) en la web. Quitamos 'gain' para no inflar tamaño.
trees = [json.loads(t) for t in booster.get_dump(dump_format='json', with_stats=True)]
def _strip_gain(n):
    n.pop('gain', None)
    for c in n.get('children', []):
        _strip_gain(c)
for t in trees:
    _strip_gain(t)

# Chequear que ya NO queden nodos categóricos (sin split_condition)
sin_sc = 0
for t in trees:
    st = [t]
    while st:
        n = st.pop()
        if 'leaf' in n:
            continue
        if 'split_condition' not in n:
            sin_sc += 1
        st.extend(n.get('children', []))
print(f'Nodos sin split_condition (debe ser 0): {sin_sc}')

# Confirmar que el modelo reentrenado ≈ el original (métricas no cambian)
p_new = np.expm1(modelo_web.predict(Xte))
p_old = np.expm1(xgb_best.predict(X_test_enc))
print(f'Diferencia media modelo reentrenado vs original: '
      f'{np.mean(np.abs(p_new - p_old)):,.0f} CLP '
      f'({np.mean(np.abs(p_new - p_old) / p_old) * 100:.3f}%)')

# 3. base_score (XGBoost 3.x lo da como texto '[1.58E1]' -> limpiar corchetes)
config = json.loads(booster.save_config())
base_score = float(config['learner']['learner_model_param']['base_score'].strip('[]'))

# 4. Orden de features
feature_names = list(Xtr.columns)

# 5. Target Encoders (Marca / Modelo -> media de log-precio)
te_marca  = {str(k): float(v) for k, v in te_maps['Marca'].items()}
te_modelo = {str(k): float(v) for k, v in te_maps['Modelo'].items()}

# 6. Marca -> lista de Modelos (para los desplegables)
marca_modelos = {
    str(m): sorted(map(str, df_model.loc[df_model['Marca'] == m, 'Modelo'].unique()))
    for m in sorted(df_model['Marca'].unique())
}

# 7. Rangos para la UI
stats = {
    'km_min': int(df_model['Kilometraje'].min()),
    'km_max': int(df_model['Kilometraje'].max()),
    'anio_min': int(df_model['Ano'].min()),
    'anio_max': int(df_model['Ano'].max()),
    'precio_mediana': float(df_model['price'].median()),
}

export = {
    'trees': trees,
    'base_score': base_score,
    'feature_names': feature_names,
    'te_maps': {'Marca': te_marca, 'Modelo': te_modelo},
    'te_globals': {'Marca': float(te_globals['Marca']),
                   'Modelo': float(te_globals['Modelo'])},
    'ohe': {
        'Combustible': {'base': 'Bencina',
                        'cols': ['Diesel', 'Eléctrico', 'Gas', 'Híbrido', 'Otro']},
        'Transmision': {'base': 'Automática', 'cols': ['Manual']},
    },
    'umbrales': {'inf': float(UMBRAL_INF), 'sup': float(UMBRAL_SUP)},
    'marca_modelos': marca_modelos,
    'stats': stats,
    'meta': {'mape': 18.6, 'r2_test': 0.816},
}

# ── Guardar el archivo ──────────────────────────────────────────────────────
with open('modelo_web.js', 'w', encoding='utf-8') as f:
    f.write('window.MODELO_DATA = ')
    json.dump(export, f, ensure_ascii=False)
    f.write(';')

import os
print(f'\n✓ modelo_web.js generado ({os.path.getsize("modelo_web.js")/1e6:.2f} MB)')
print(f'  Árboles: {len(trees)} | Marcas: {len(marca_modelos)} | Modelos: {len(te_modelo)}')
print(f'  Umbrales: P25={UMBRAL_INF:+.1f}%  P75={UMBRAL_SUP:+.1f}%')

# ── VERIFICACIÓN: la fórmula JS debe coincidir con predict() ────────────────
def _manual_pred(row):
    s = base_score
    for t in trees:
        node = t
        while 'leaf' not in node:
            v = row[node['split']]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                nid = node['missing']
            else:
                nid = node['yes'] if v < node['split_condition'] else node['no']
            node = next(c for c in node['children'] if c['nodeid'] == nid)
        s += node['leaf']
    return s

muestra = Xtr.head(5).to_dict('records')
manual  = np.array([_manual_pred(r) for r in muestra])
oficial = modelo_web.predict(Xtr.head(5))
print('\nVerificación (JS vs XGBoost — deben ser casi idénticos):')
for a, b in zip(manual, oficial):
    print(f'  manual={a:.5f}  xgboost={b:.5f}  Δ={abs(a-b):.2e}')

from google.colab import files
files.download('modelo_web.js')
