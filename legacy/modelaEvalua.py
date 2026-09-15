# modelaEvalua.py - Versión completa con Stepwise implementado
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("\n--- 4. Modelado ---")

# Diccionario para almacenar resultados de los modelos
results = {}

# --- 4.1 Regresión Lineal MCO (statsmodels para diagnóstico) ---
print("\n--- 4.1 Modelo: Regresión Lineal MCO (statsmodels) ---")
# Añadir constante (intercepto) para statsmodels
X_train_sm = sm.add_constant(X_train_proc)
X_test_sm = sm.add_constant(X_test_proc)

try:
    ols_model = sm.OLS(y_train, X_train_sm).fit()
    print(ols_model.summary(xname=['const'] + feature_names)) # Pasar nombres de columnas

    # --- Diagnóstico OLS ---
    print("\nDiagnóstico del Modelo OLS:")
    # 1. Linealidad: Gráfico Residuos vs Ajustados
    y_pred_train_ols = ols_model.predict(X_train_sm)
    residuals_ols = y_train - y_pred_train_ols
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred_train_ols, y=residuals_ols)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Valores Ajustados (Entrenamiento)')
    plt.ylabel('Residuos')
    plt.title('Gráfico Residuos vs Ajustados (OLS)')
    plt.grid(True)
    plt.show()

    # 2. Normalidad de Residuos: Q-Q plot
    plt.figure(figsize=(8, 5))
    sm.qqplot(residuals_ols, line='s')
    plt.title('Q-Q Plot de Residuos (OLS)')
    plt.grid(True)
    plt.show()
    print(f"Test Jarque-Bera (Normalidad Residuos): JB={ols_model.jarque_bera[0]:.2f}, p={ols_model.jarque_bera[1]:.3f}")

    # 3. Independencia de Errores
    print(f"Test Durbin-Watson (Autocorrelación Residuos): DW={ols_model.durbin_watson:.2f}")

    # Evaluación en Test
    y_pred_test_ols = ols_model.predict(X_test_sm)
    mse_ols = mean_squared_error(y_test, y_pred_test_ols)
    r2_ols = r2_score(y_test, y_pred_test_ols)
    mae_ols = mean_absolute_error(y_test, y_pred_test_ols)
    results['OLS_Statsmodels'] = {'MSE': mse_ols, 'R2': r2_ols, 'MAE': mae_ols, 'Model': ols_model}
    print(f"\nEvaluación OLS (Statsmodels) en Test: MSE={mse_ols:.4f}, R2={r2_ols:.4f}, MAE={mae_ols:.4f}")

except np.linalg.LinAlgError:
    print("Error: Problema de multicolinealidad detectado en OLS (statsmodels).")
    print("Considera usar modelos regularizados o eliminar variables correlacionadas.")
    results['OLS_Statsmodels'] = {'MSE': np.inf, 'R2': -np.inf, 'MAE': np.inf, 'Model': None}
except Exception as e:
    print(f"Error inesperado durante el ajuste OLS (statsmodels): {e}")
    results['OLS_Statsmodels'] = {'MSE': np.inf, 'R2': -np.inf, 'MAE': np.inf, 'Model': None}

# --- 4.2 Método Stepwise (AIC) - Implementación Completa ---
print("\n--- 4.2 Modelo: Regresión Stepwise (AIC) ---")

try:
    # Implementación con SequentialFeatureSelector de scikit-learn
    sfs = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select='auto',  # Selección automática basada en criterios
        direction='forward',          # Stepwise forward
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1
    ).fit(X_train_proc, y_train)

    # Obtener características seleccionadas
    selected_features = X_train_proc.columns[sfs.get_support()].tolist()
    print(f"Características seleccionadas por Stepwise ({len(selected_features)}): {selected_features}")

    # Entrenar modelo final con características seleccionadas
    lr_stepwise = LinearRegression()
    lr_stepwise.fit(X_train_proc[selected_features], y_train)

    # Calcular AIC manualmente (para comparación con otros modelos)
    n = len(y_train)
    k = len(selected_features) + 1  # +1 para el intercepto
    y_pred_train_step = lr_stepwise.predict(X_train_proc[selected_features])
    rss = np.sum((y_train - y_pred_train_step)**2)
    aic_stepwise = n * np.log(rss/n) + 2*k
    
    # Evaluación en Test
    y_pred_test_step = lr_stepwise.predict(X_test_proc[selected_features])
    mse_stepwise = mean_squared_error(y_test, y_pred_test_step)
    r2_stepwise = r2_score(y_test, y_pred_test_step)
    mae_stepwise = mean_absolute_error(y_test, y_pred_test_step)
    
    results['Stepwise'] = {
        'MSE': mse_stepwise,
        'R2': r2_stepwise,
        'MAE': mae_stepwise,
        'AIC': aic_stepwise,
        'Model': lr_stepwise,
        'Features': selected_features
    }
    print(f"\nEvaluación Stepwise en Test: MSE={mse_stepwise:.4f}, R2={r2_stepwise:.4f}, MAE={mae_stepwise:.4f}, AIC={aic_stepwise:.2f}")

except Exception as e:
    print(f"Error durante el proceso Stepwise: {e}")
    results['Stepwise'] = {'MSE': np.nan, 'R2': np.nan, 'MAE': np.nan, 'AIC': np.nan, 'Model': None, 'Features': None}

# --- 4.3 Regresión Ridge (con CV para lambda óptimo) ---
print("\n--- 4.3 Modelo: Regresión Ridge (con CV) ---")

alphas_ridge = np.logspace(-6, 6, 13)  # 1e-6 a 1e6
ridge_cv = RidgeCV(alphas=alphas_ridge, store_cv_values=True, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train_proc, y_train)

optimal_alpha_ridge = ridge_cv.alpha_
print(f"Alpha (lambda) óptimo encontrado para Ridge: {optimal_alpha_ridge:.6f}")

# Entrenar modelo final Ridge con alpha óptimo
ridge_final = Ridge(alpha=optimal_alpha_ridge)
ridge_final.fit(X_train_proc, y_train)

# Evaluación en Test
y_pred_test_ridge = ridge_final.predict(X_test_proc)
mse_ridge = mean_squared_error(y_test, y_pred_test_ridge)
r2_ridge = r2_score(y_test, y_pred_test_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_test_ridge)
results['Ridge'] = {'MSE': mse_ridge, 'R2': r2_ridge, 'MAE': mae_ridge, 'Model': ridge_final, 'Alpha': optimal_alpha_ridge}
print(f"Evaluación Ridge (alpha={optimal_alpha_ridge:.6f}) en Test: MSE={mse_ridge:.4f}, R2={r2_ridge:.4f}, MAE={mae_ridge:.4f}")

# --- 4.4 Regresión Lasso (con CV para lambda óptimo) ---
print("\n--- 4.4 Modelo: Regresión Lasso (con CV) ---")

lasso_cv = LassoCV(cv=5, random_state=SEED, n_jobs=-1, max_iter=10000)
lasso_cv.fit(X_train_proc, y_train)

optimal_alpha_lasso = lasso_cv.alpha_
print(f"Alpha (lambda) óptimo encontrado para Lasso: {optimal_alpha_lasso:.6f}")

# Modelo ajustado ya usa el alpha óptimo
lasso_final = lasso_cv

# Evaluación en Test
y_pred_test_lasso = lasso_final.predict(X_test_proc)
mse_lasso = mean_squared_error(y_test, y_pred_test_lasso)
r2_lasso = r2_score(y_test, y_pred_test_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_test_lasso)

# Variables seleccionadas (coeficientes no cero)
coefs_lasso = pd.Series(lasso_final.coef_, index=feature_names)
selected_lasso_features = coefs_lasso[coefs_lasso != 0].index.tolist()

results['Lasso'] = {
    'MSE': mse_lasso,
    'R2': r2_lasso,
    'MAE': mae_lasso,
    'Model': lasso_final,
    'Alpha': optimal_alpha_lasso,
    'Features': selected_lasso_features
}
print(f"Evaluación Lasso (alpha={optimal_alpha_lasso:.6f}) en Test: MSE={mse_lasso:.4f}, R2={r2_lasso:.4f}, MAE={mae_lasso:.4f}")
print(f"Número de características seleccionadas por Lasso: {len(selected_lasso_features)}")

# --- 4.5 Regresión Elastic Net (con CV para lambda y alpha óptimos) ---
print("\n--- 4.5 Modelo: Regresión Elastic Net (con CV) ---")

l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
elastic_net_cv = ElasticNetCV(
    l1_ratio=l1_ratios,
    cv=5,
    random_state=SEED,
    n_jobs=-1,
    max_iter=10000
)
elastic_net_cv.fit(X_train_proc, y_train)

optimal_alpha_enet = elastic_net_cv.alpha_
optimal_l1_ratio_enet = elastic_net_cv.l1_ratio_
print(f"Alpha (lambda) óptimo encontrado para ElasticNet: {optimal_alpha_enet:.6f}")
print(f"L1 Ratio (alpha mezcla) óptimo encontrado para ElasticNet: {optimal_l1_ratio_enet:.2f}")

# Modelo ajustado ya usa los parámetros óptimos
elastic_net_final = elastic_net_cv

# Evaluación en Test
y_pred_test_enet = elastic_net_final.predict(X_test_proc)
mse_enet = mean_squared_error(y_test, y_pred_test_enet)
r2_enet = r2_score(y_test, y_pred_test_enet)
mae_enet = mean_absolute_error(y_test, y_pred_test_enet)

# Variables seleccionadas
coefs_enet = pd.Series(elastic_net_final.coef_, index=feature_names)
selected_enet_features = coefs_enet[coefs_enet != 0].index.tolist()

results['ElasticNet'] = {
    'MSE': mse_enet,
    'R2': r2_enet,
    'MAE': mae_enet,
    'Model': elastic_net_final,
    'Alpha': optimal_alpha_enet,
    'L1_Ratio': optimal_l1_ratio_enet,
    'Features': selected_enet_features
}
print(f"Evaluación ElasticNet (alpha={optimal_alpha_enet:.6f}, l1_ratio={optimal_l1_ratio_enet:.2f}) en Test: MSE={mse_enet:.4f}, R2={r2_enet:.4f}, MAE={mae_enet:.4f}")
print(f"Número de características seleccionadas por ElasticNet: {len(selected_enet_features)}")

# --- 5. Comparación de Modelos y Selección ---
print("\n--- 5. Comparación de Resultados en Conjunto de Prueba ---")

# Crear DataFrame comparativo
comparison_df = pd.DataFrame({
    'Modelo': list(results.keys()),
    'MSE': [res.get('MSE', np.nan) for res in results.values()],
    'R2': [res.get('R2', np.nan) for res in results.values()],
    'MAE': [res.get('MAE', np.nan) for res in results.values()],
    'AIC': [res.get('AIC', np.nan) for res in results.values()],
    'N_Features': [
        len(res['Features']) if 'Features' in res else 
        (res['Model'].coef_ != 0).sum() if hasattr(res.get('Model', None), 'coef_') else 
        len(feature_names) if res.get('Model', None) is not None else np.nan
        for res in results.values()
    ],
    'Parametros': [
        f"α={res.get('Alpha', '--'):.4f}" + (f", l1={res.get('L1_Ratio', '--')}" if 'L1_Ratio' in res else "") 
        for res in results.values()
    ]
})

# Ordenar por R2 (mayor es mejor)
comparison_df.sort_values(by='R2', ascending=False, inplace=True)
print("\nComparación de Modelos (ordenados por R2):")
print(comparison_df.to_string())

# --- Selección del Mejor Modelo ---
best_model_name = comparison_df.iloc[0]['Modelo']
best_model_details = results[best_model_name]
print(f"\nModelo con mejor rendimiento: {best_model_name}")
print(f"  - R2: {best_model_details['R2']:.4f}")
print(f"  - MSE: {best_model_details['MSE']:.4f}")
print(f"  - MAE: {best_model_details['MAE']:.4f}")
if 'AIC' in best_model_details:
    print(f"  - AIC: {best_model_details['AIC']:.2f}")

# Mostrar variables importantes del mejor modelo
if 'Features' in best_model_details and best_model_details['Features']:
    print(f"\nVariables seleccionadas por {best_model_name} ({len(best_model_details['Features'])}):")
    print(best_model_details['Features'])
elif hasattr(best_model_details.get('Model', None), 'coef_'):
    coefs = pd.Series(best_model_details['Model'].coef_, index=feature_names)
    top_features = coefs.abs().sort_values(ascending=False).head(15)
    print("\nTop 15 variables más importantes (coeficientes absolutos):")
    print(top_features)

print("\n--- Fin del Modelado y Evaluación ---")