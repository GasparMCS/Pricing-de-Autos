import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
    df['Combustible'] = df['Combustible'].apply(limpiar_combustible)
    if 'Transmision' in df.columns:
        df['Transmision'] = df['Transmision'].apply(limpiar_transmision)
        df['Transmision'] = df['Transmision'].fillna(df['Transmision'].mode()[0])
    df['Combustible'] = df['Combustible'].fillna(df['Combustible'].mode()[0])
    df = df.dropna(subset=['Marca','Modelo','Ano','Kilometraje','price'])
    for col in ['price','Kilometraje']:
        Q1,Q3 = df[col].quantile(.25),df[col].quantile(.75); IQR=Q3-Q1
        df = df[(df[col]>=Q1-1.5*IQR)&(df[col]<=Q3+1.5*IQR)]
    df = df[(df['price']>500000)&(df['Ano']>=1990)&(df['Ano']<=2026)&(df['Kilometraje']>0)]
    df['antiguedad_auto'] = 2026 - df['Ano']
    df['price_log'] = np.log1p(df['price'])
    df['Marca'] = df['Marca'].str.strip().str.title()
    df['Modelo'] = df['Modelo'].str.strip().str.title()
    return df.reset_index(drop=True)

df = clean_dataset(pd.read_csv(BASE/'datos_combinados_entrega2.csv'))

np.random.seed(42)
df['_t'] = pd.qcut(df['price'], q=3, labels=False, duplicates='drop')
demo_idx = []
for t in range(3):
    sub = df[df['_t']==t]
    demo_idx.extend(sub.sample(n=min(5,len(sub)), random_state=42).index.tolist())
df_model = df.drop(index=demo_idx).drop(columns='_t').reset_index(drop=True)

X = df_model[['antiguedad_auto','Kilometraje','Combustible','Transmision','Marca','Modelo','price']].copy()
y_log = df_model['price_log']
bins = pd.qcut(df_model['price'], q=5, labels=False, duplicates='drop')
X_train,X_test,y_tr,y_te = train_test_split(X, y_log, test_size=0.2, random_state=42, stratify=bins)

def te_fit(s, t, sm=10):
    gm = t.mean(); st = t.groupby(s).agg(['mean','count'])
    lam = st['count']/(st['count']+sm)
    return (lam*st['mean']+(1-lam)*gm).to_dict(), gm

te_maps, te_glo = {}, {}
for c in ['Marca','Modelo']: te_maps[c], te_glo[c] = te_fit(X_train[c], y_tr)

ohe_tr = pd.get_dummies(X_train[['Combustible','Transmision']], prefix=['Combustible','Transmision'], drop_first=True)

X_enc = pd.DataFrame({
    'antiguedad_auto': X_train['antiguedad_auto'].values,
    'Kilometraje':     X_train['Kilometraje'].values,
    'Marca_te':        X_train['Marca'].map(te_maps['Marca']).fillna(te_glo['Marca']).values,
    'Modelo_te':       X_train['Modelo'].map(te_maps['Modelo']).fillna(te_glo['Modelo']).values,
})
X_enc = pd.concat([X_enc.reset_index(drop=True), ohe_tr.reset_index(drop=True)], axis=1)
X_enc = X_enc.astype(float)  # convertir booleanos a float para VIF

print('=== MATRIZ DE CORRELACION ===')
corr = X_enc.corr().round(2)
print(corr.to_string())

print()
print('=== VIF (todas las variables del modelo) ===')
X_vif = sm.add_constant(X_enc)
vif_data = pd.DataFrame({
    'Feature': X_enc.columns,
    'VIF': [round(variance_inflation_factor(X_vif.values, i+1), 2) for i in range(X_enc.shape[1])]
}).sort_values('VIF', ascending=False)
print(vif_data.to_string(index=False))
print()
print('Referencia: VIF < 5 = sin problema | 5-10 = moderado | > 10 = critico')
