import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

print('Cargando y limpiando datos...')
df = clean_dataset(pd.read_csv(BASE/'datos_combinados_entrega2.csv'))
print(f'Dataset: {len(df):,} registros')

# Autos demo (extraidos antes de entrenar)
np.random.seed(42)
df['_t'] = pd.qcut(df['price'], q=3, labels=False, duplicates='drop')
demo_idx = []
for t in range(3):
    sub = df[df['_t']==t]
    demo_idx.extend(sub.sample(n=min(5,len(sub)), random_state=42).index.tolist())
df_demo = df.loc[demo_idx].copy()
df_model = df.drop(index=demo_idx).drop(columns='_t').reset_index(drop=True)
print(f'Demo: {len(df_demo)} autos | Modelo: {len(df_model):,} autos')

NUM = ['antiguedad_auto','Kilometraje']
CAT_OHE = ['Combustible','Transmision']
CAT_TE = ['Marca','Modelo']

X = df_model[NUM+CAT_OHE+CAT_TE+['price']].copy()
y_log = df_model['price_log']
y_orig = df_model['price']

bins = pd.qcut(y_orig, q=5, labels=False, duplicates='drop')
X_train,X_test,y_tr,y_te = train_test_split(X, y_log, test_size=0.2, random_state=42, stratify=bins)
print(f'Train: {len(X_train):,} | Test: {len(X_test):,}')

# Target encoding (solo en train)
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

def mape(yt, yp):
    yt,yp = np.array(yt),np.array(yp); m = yt!=0
    return np.mean(np.abs((yt[m]-yp[m])/yt[m]))*100

def evaluate(model, name):
    ptr = model.predict(Xtr); pte = model.predict(Xte)
    ytr_o = np.expm1(y_tr); yte_o = np.expm1(y_te)
    ptr_o = np.expm1(ptr);  pte_o = np.expm1(pte)
    r2tr = r2_score(y_tr, ptr); r2te = r2_score(y_te, pte)
    mae  = mean_absolute_error(yte_o, pte_o)/1e6
    mp   = mape(yte_o, pte_o)
    print(f'  {name}: R2_train={r2tr:.3f} | R2_test={r2te:.3f} | GAP={r2tr-r2te:.3f} | MAE=${mae:.2f}M | MAPE={mp:.1f}%')
    return {'Modelo':name,'R2_train':round(r2tr,3),'R2_test':round(r2te,3),
            'GAP':round(r2tr-r2te,3),'MAE_M':round(mae,2),'MAPE':round(mp,1)}, np.expm1(pte)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

# Baseline
bm  = X_train.groupby('Marca')['price'].median()
gm_b = X_train['price'].median()
bp  = X_test['Marca'].map(bm).fillna(gm_b)
yte_o = np.expm1(y_te)
base = {'Modelo':'Baseline','R2_train':0,'R2_test':round(r2_score(yte_o,bp),3),'GAP':0,
        'MAE_M':round(mean_absolute_error(yte_o,bp)/1e6,2),'MAPE':round(mape(yte_o,bp),1)}
print(f'  Baseline: R2_test={base["R2_test"]} | MAE=${base["MAE_M"]}M | MAPE={base["MAPE"]}%')
results.append(base)

# Ridge
print('Entrenando Ridge...')
sc = StandardScaler()
Xtr_sc = sc.fit_transform(Xtr); Xte_sc = sc.transform(Xte)
rgs = GridSearchCV(Ridge(), {'alpha':[1,10,50,100,500]}, cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1)
rgs.fit(Xtr_sc, y_tr)
print(f'  Ridge best alpha: {rgs.best_params_}')
class ScM:
    def __init__(self,m,s): self.m=m; self.s=s
    def predict(self,X): return self.m.predict(self.s.transform(X))
rm, rp = evaluate(ScM(rgs.best_estimator_, sc), 'Ridge')
results.append(rm)

# Random Forest
print('Entrenando Random Forest...')
rfgs = GridSearchCV(RandomForestRegressor(random_state=42),
                    {'n_estimators':[100,200],'max_depth':[10,20],'min_samples_leaf':[1,5]},
                    cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
rfgs.fit(Xtr, y_tr)
print(f'  RF best: {rfgs.best_params_}')
rfm, rfp = evaluate(rfgs.best_estimator_, 'Random Forest')
results.append(rfm)

# Lasso
print('Entrenando Lasso...')
from sklearn.linear_model import Lasso
lgs = GridSearchCV(Lasso(max_iter=10000), {'alpha':[0.0001,0.001,0.01,0.1,1]}, cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1)
lgs.fit(Xtr_sc, y_tr)
print(f'  Lasso best alpha: {lgs.best_params_}')
lm, lp = evaluate(ScM(lgs.best_estimator_, sc), 'Lasso')
results.append(lm)

# Coeficientes Lasso — ver qué variables eliminó
lasso_coef = pd.Series(lgs.best_estimator_.coef_, index=Xtr.columns).sort_values(key=abs, ascending=False)
print()
print('  Coeficientes Lasso (0 = variable eliminada):')
for feat, coef in lasso_coef.items():
    status = 'ELIMINADA' if coef == 0 else ''
    print(f'    {feat:<30} {coef:+.4f}  {status}')

# XGBoost
try:
    import xgboost as xgb
    print('Entrenando XGBoost...')
    xgs = GridSearchCV(xgb.XGBRegressor(random_state=42, verbosity=0),
                       {'n_estimators':[100,300],'max_depth':[3,5],'learning_rate':[0.05,0.1],'subsample':[0.8,1.0]},
                       cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
    xgs.fit(Xtr, y_tr)
    print(f'  XGB best: {xgs.best_params_}')
    xm, xp = evaluate(xgs.best_estimator_, 'XGBoost')
    results.append(xm)
except ImportError:
    print('XGBoost no disponible')

print()
print('='*65)
print('TABLA COMPARATIVA DE MODELOS')
print('='*65)
print(pd.DataFrame(results).to_string(index=False))
