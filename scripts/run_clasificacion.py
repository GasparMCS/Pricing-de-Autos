import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

BASE = Path(r'C:\Users\gcamp\Downloads\UDP\Proyecto DataSciense')

def limpiar_combustible(v):
    if pd.isna(v): return np.nan
    v = str(v).lower()
    if 'diesel' in v or 'petroleo' in v: return 'Diesel'
    elif 'bencina' in v or 'gasolina' in v or 'gasoline' in v or 'petrol' in v: return 'Bencina'
    elif 'hibrido' in v or 'hybrid' in v: return 'Hibrido'
    elif 'electrico' in v or 'electric' in v: return 'Electrico'
    elif 'gas' in v: return 'Gas'
    return 'Otro'

def limpiar_transmision(v):
    if pd.isna(v): return np.nan
    v = str(v).lower().strip()
    if v in ['m','manual','mecanica','mechanical']: return 'Manual'
    elif 'auto' in v or 'cvt' in v or 'tiptronic' in v: return 'Automatica'
    return np.nan

def clean_dataset(df_raw):
    df = df_raw.copy()
    col_map = {'make':'Marca','model':'Modelo','year':'Ano','km':'Kilometraje',
               'price_clp':'price','fuel_type':'Combustible','transmission':'Transmision'}
    df.rename(columns={k:v for k,v in col_map.items() if k in df.columns}, inplace=True)
    for col in ['Ano','Kilometraje','price']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Combustible'] = df['Combustible'].apply(limpiar_combustible) if 'Combustible' in df.columns else 'Bencina'
    if 'Transmision' in df.columns:
        df['Transmision'] = df['Transmision'].apply(limpiar_transmision)
        df['Transmision'] = df['Transmision'].fillna(df['Transmision'].mode()[0])
    df['Combustible'] = df['Combustible'].fillna(df['Combustible'].mode()[0])
    df = df.dropna(subset=['Marca','Modelo','Ano','Kilometraje','price'])
    for col in ['price','Kilometraje']:
        Q1,Q3 = df[col].quantile(.25), df[col].quantile(.75); IQR = Q3-Q1
        df = df[(df[col] >= Q1-1.5*IQR) & (df[col] <= Q3+1.5*IQR)]
    df = df[(df['price']>500000)&(df['Ano']>=1990)&(df['Ano']<=2026)&(df['Kilometraje']>0)]
    df['antiguedad_auto'] = 2026 - df['Ano']
    df['price_log'] = np.log1p(df['price'])
    df['Marca'] = df['Marca'].str.strip().str.title()
    df['Modelo'] = df['Modelo'].str.strip().str.title()
    return df.reset_index(drop=True)

# ── Cargar y limpiar ───────────────────────────────────────────────────────
df = clean_dataset(pd.read_csv(BASE/'datos_combinados_entrega2.csv'))

# ── Extraer autos demo ANTES de entrenar ──────────────────────────────────
np.random.seed(42)
df['_t'] = pd.qcut(df['price'], q=3, labels=False, duplicates='drop')
demo_idx = []
for t in range(3):
    sub = df[df['_t']==t]
    demo_idx.extend(sub.sample(n=min(5,len(sub)), random_state=42).index.tolist())
df_demo  = df.loc[demo_idx].copy().reset_index(drop=True)
df_model = df.drop(index=demo_idx).drop(columns='_t').reset_index(drop=True)

NUM     = ['antiguedad_auto','Kilometraje']
CAT_OHE = ['Combustible','Transmision']
CAT_TE  = ['Marca','Modelo']

X      = df_model[NUM+CAT_OHE+CAT_TE+['price']].copy()
y_log  = df_model['price_log']
y_orig = df_model['price']

bins = pd.qcut(y_orig, q=5, labels=False, duplicates='drop')
X_train,X_test,y_tr,y_te = train_test_split(X, y_log, test_size=0.2, random_state=42, stratify=bins)

# Target encoding solo en train
def te_fit(s, t, sm=10):
    gm = t.mean()
    st = t.groupby(s).agg(['mean','count'])
    lam = st['count']/(st['count']+sm)
    return (lam*st['mean']+(1-lam)*gm).to_dict(), gm

te_maps, te_glo = {}, {}
for c in CAT_TE: te_maps[c], te_glo[c] = te_fit(X_train[c], y_tr)

ohe_tr = pd.get_dummies(X_train[CAT_OHE], prefix=CAT_OHE, drop_first=True)
ohe_te = pd.get_dummies(X_test[CAT_OHE],  prefix=CAT_OHE, drop_first=True).reindex(columns=ohe_tr.columns, fill_value=0)

def enc(X_sp, ohe):
    num = X_sp[NUM].reset_index(drop=True)
    te  = pd.DataFrame({c+'_te': pd.Series(X_sp[c].values).map(te_maps[c]).fillna(te_glo[c]) for c in CAT_TE})
    return pd.concat([num, te, ohe.reset_index(drop=True)], axis=1)

Xtr = enc(X_train, ohe_tr)
Xte = enc(X_test,  ohe_te)

# ── Entrenar XGBoost (mejores params encontrados antes) ───────────────────
print('Entrenando XGBoost...')
model = xgb.XGBRegressor(
    learning_rate=0.1, max_depth=5, n_estimators=300,
    subsample=1.0, random_state=42, verbosity=0
)
model.fit(Xtr, y_tr)
print('Listo.')

# ── Clasificacion: autos demo (nunca vistos por el modelo) ────────────────
print()
print('='*65)
print('CLASIFICACION DE AUTOS DE DEMOSTRACION')
print('Estos 15 autos NUNCA fueron vistos por el modelo')
print('Comparamos: precio publicado vs precio predicho por XGBoost')
print('='*65)

# Preparar features de los autos demo con los mismos encodings del train
ohe_demo = pd.get_dummies(df_demo[CAT_OHE], prefix=CAT_OHE, drop_first=True).reindex(columns=ohe_tr.columns, fill_value=0)
X_demo = enc(df_demo[NUM+CAT_OHE+CAT_TE], ohe_demo)

pred_log  = model.predict(X_demo)
precio_predicho = np.expm1(pred_log)

df_result = df_demo[['Marca','Modelo','Ano','Kilometraje','Combustible','price']].copy()
df_result['precio_predicho'] = precio_predicho.round(0).astype(int)
df_result['diferencia_pct']  = ((df_result['price'] - df_result['precio_predicho'])
                                 / df_result['precio_predicho'] * 100).round(1)

# Umbral basado en MAPE del modelo (18.8% / 2 ~ 15%)
UMBRAL = 15.0

def clasificar(pct):
    if pct > UMBRAL:   return 'SOBREVALORADO'
    elif pct < -UMBRAL: return 'SUBVALORADO'
    else:               return 'PRECIO JUSTO'

df_result['clasificacion'] = df_result['diferencia_pct'].apply(clasificar)

# Mostrar resultado
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
for _, row in df_result.iterrows():
    signo = '+' if row['diferencia_pct'] > 0 else ''
    emoji = {'SOBREVALORADO':'ROJO  ', 'SUBVALORADO':'VERDE ', 'PRECIO JUSTO':'AMARILLO'}[row['clasificacion']]
    print(f"{emoji} {row['Marca']:<12} {row['Modelo']:<15} {int(row['Ano'])}  "
          f"Publicado: ${row['price']:>10,.0f}  "
          f"Predicho: ${row['precio_predicho']:>10,.0f}  "
          f"Diferencia: {signo}{row['diferencia_pct']:.1f}%  "
          f"-> {row['clasificacion']}")

print()
print('Distribución:')
print(df_result['clasificacion'].value_counts().to_string())
print()
print(f'Umbral usado: +-{UMBRAL}% (basado en MAPE del modelo: 18.8%)')
print()
print('LOGICA: precio_publicado > precio_predicho => el vendedor pide MAS de lo que vale => SOBREVALORADO')
print('        precio_publicado < precio_predicho => el vendedor pide MENOS de lo que vale => SUBVALORADO')
