"""
Proyecto: Modelado del Consumo de ClO2 en Proceso de Blanqueo de Celulosa - Versión Excel

Descripción:
Este script implementa un análisis completo para predecir el consumo de dióxido de cloro
en el proceso de blanqueo de celulosa, utilizando diferentes técnicas de regresión.
Versión modificada para leer directamente de archivo Excel (.xlsx).

Pasos principales:
1. Carga y exploración inicial de datos desde Excel
2. Preparación y limpieza de datos
3. Modelado con diferentes técnicas
4. Evaluación y selección del mejor modelo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ==============================================================================
# --- Configuración Inicial ---
# ==============================================================================
print("=== INICIO DEL ANÁLISIS ===")

# Configuración de parámetros según lo solicitado en el proyecto
CONFIG = {
    'file_path': 'Ab19selec.xlsx',       # Archivo de datos en formato Excel
    'target_variable': 'Consumo Total ClO2',  # Variable objetivo según el problema
    'SEED': 2022,                        # Semilla para reproducibilidad
    'TEST_SIZE': 0.20,                   # Proporción para conjunto de prueba (80-20)
    'sheet_name': 'Hoja1',               # Nombre de la hoja en el archivo Excel
    'threshold_variance': 0.01,          # Umbral para eliminar características con baja varianza
    'correlation_threshold': 0.95        # Umbral para eliminar características altamente correlacionadas
}

print("\n--- Configuración Inicial ---")
print(f"Semilla aleatoria: {CONFIG['SEED']}")
print(f"Tamaño del conjunto de prueba: {CONFIG['TEST_SIZE']*100}%")
print(f"Variable objetivo: '{CONFIG['target_variable']}'")
print(f"Archivo de datos: {CONFIG['file_path']}")
print(f"Hoja Excel: '{CONFIG['sheet_name']}'")

# Configurar semilla para reproducibilidad
np.random.seed(CONFIG['SEED'])

# ==============================================================================
# --- 1. Carga y Exploración de Datos (Desde Excel) ---
# ==============================================================================
print("\n=== 1. Carga y Exploración de Datos ===")

def cargar_datos():
    """Carga y realiza exploración inicial del dataset desde archivo Excel"""
    try:
        # Cargar datos desde archivo Excel
        df = pd.read_excel(
            CONFIG['file_path'],
            sheet_name=CONFIG['sheet_name'],
            parse_dates=[0]             # Intentar parsear la primera columna como fecha
        )
        
        # Eliminar filas completamente vacías si las hay
        df = df.dropna(how='all')
        
        # Forzar conversión de variable objetivo a numérico
        df[CONFIG['target_variable']] = pd.to_numeric(df[CONFIG['target_variable']], errors='coerce')
        
        # Eliminar filas donde la variable objetivo no pudo convertirse a numérico
        df = df.dropna(subset=[CONFIG['target_variable']])
        
        print(f"\nDatos cargados exitosamente. Dimensiones: {df.shape}")
        print(f"Columnas disponibles: {list(df.columns)}")
        
        # Verificar si la primera columna es datetime y establecer como índice
        if pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
            df.set_index(df.columns[0], inplace=True)
            print(f"\nColumna '{df.index.name}' establecida como índice (datetime)")
        else:
            print("\nAdvertencia: La primera columna no es datetime - se mantendrá como índice simple")
            df.set_index(df.columns[0], inplace=True)
        
        # Verificar variable objetivo
        if CONFIG['target_variable'] not in df.columns:
            print(f"\nError: Variable objetivo '{CONFIG['target_variable']}' no encontrada")
            print("Columnas disponibles:", df.columns.tolist())
            exit()
        else:
            print(f"\nVariable objetivo '{CONFIG['target_variable']}' encontrada")
            print(f"Tipo de dato de la variable objetivo: {df[CONFIG['target_variable']].dtype}")
        
        return df
    
    except FileNotFoundError:
        print(f"\nError: Archivo no encontrado en la ruta: {CONFIG['file_path']}")
        print("Por favor verifique que:")
        print("1. El archivo existe en la ubicación especificada")
        print("2. El nombre del archivo es correcto (incluyendo la extensión .xlsx)")
        exit()
    except Exception as e:
        print(f"\nError al cargar los datos desde Excel: {str(e)}")
        print("Posibles soluciones:")
        print("1. Verificar que el archivo no esté corrupto")
        print("2. Verificar que la hoja especificada exista en el archivo")
        print("3. Probar con openpyxl como motor de lectura: pd.read_excel(..., engine='openpyxl')")
        exit()

def explorar_datos(df):
    """Realiza análisis exploratorio de los datos"""
    print("\n--- Análisis Exploratorio ---")
    
    # Información general del DataFrame
    print("\nInformación del DataFrame:")
    df.info()
    
    # Estadísticas descriptivas para variables numéricas
    print("\nEstadísticas descriptivas (variables numéricas):")
    numeric_cols = df.select_dtypes(include=np.number).columns
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df[numeric_cols].describe().transpose())
    
    # Valores faltantes
    print("\nValores faltantes por columna:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    # Columnas con varianza cero (constantes)
    print("\nColumnas con varianza cero (constantes):")
    if len(numeric_cols) > 0:
        constant_cols = df[numeric_cols].columns[df[numeric_cols].var() == 0].tolist()
        print(constant_cols if constant_cols else "No hay columnas constantes")
    else:
        print("No hay columnas numéricas para calcular varianza")
    
    # Visualizaciones exploratorias
    print("\nGenerando visualizaciones exploratorias...")
    
    # Distribución de la variable objetivo
    plt.figure(figsize=(12, 6))
    sns.histplot(df[CONFIG['target_variable']].dropna(), kde=True, bins=30)
    plt.title(f'Distribución de {CONFIG["target_variable"]}')
    plt.xlabel(CONFIG['target_variable'])
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Serie temporal si el índice es datetime
    if isinstance(df.index, pd.DatetimeIndex):
        plt.figure(figsize=(15, 7))
        df[CONFIG['target_variable']].plot()
        plt.title(f'Serie Temporal de {CONFIG["target_variable"]}')
        plt.ylabel(CONFIG['target_variable'])
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    # Matriz de correlación (solo para variables numéricas)
    if len(numeric_cols) > 1:
        plt.figure(figsize=(15, 12))
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                   annot=False, fmt=".2f", 
                   vmin=-1, vmax=1)
        plt.title('Matriz de Correlación (Variables Numéricas)')
        plt.tight_layout()
        plt.show()
        
        # Identificar correlaciones altas con la variable objetivo
        target_corr = corr_matrix[CONFIG['target_variable']].sort_values(
            key=abs, ascending=False)
        print("\nCorrelación con la variable objetivo:")
        print(target_corr.head(10))
    
    return df

# Cargar y explorar datos
df = cargar_datos()
df = explorar_datos(df)

# ==============================================================================
# --- 2. Preparación de Datos ---
# ==============================================================================
print("\n=== 2. Preparación de Datos ===")

def preparar_datos(df):
    """Preprocesa los datos para modelado"""
    print("\n--- Preparación de Datos ---")
    
    # 1. Eliminar columnas problemáticas (basado en análisis exploratorio)
    print("\nEliminando columnas problemáticas...")
    
    # Columnas con varianza cero (constantes)
    numeric_cols = df.select_dtypes(include=np.number).columns
    constant_cols = df[numeric_cols].columns[df[numeric_cols].var() < CONFIG['threshold_variance']].tolist()
    
    # Columnas adicionales a eliminar (basado en conocimiento del dominio)
    cols_to_drop = constant_cols + [
        col for col in df.columns 
        if any(x in str(col) for x in ['CE05_IP21_547AI1011', 'CE05_IP21_547AI1137', 
                                      'CE05_IP21_547AI1250', 'CE05_IP21_547CI1772', 
                                      'CE05_IP21_547FI1019'])
    ]
    cols_to_drop = list(set(cols_to_drop))  # Eliminar duplicados
    
    if cols_to_drop:
        print(f"Eliminando {len(cols_to_drop)} columnas problemáticas:")
        print(cols_to_drop)
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    else:
        print("No se encontraron columnas problemáticas para eliminar")
    
    # 2. Manejo de valores faltantes
    print("\nManejando valores faltantes...")
    missing_before = df.isnull().sum().sum()
    print(f"Valores faltantes iniciales: {missing_before}")
    
    # Imputación con mediana para variables numéricas (excepto target)
    numeric_cols = df.select_dtypes(include=np.number).columns
    numeric_cols_to_impute = numeric_cols.drop(CONFIG['target_variable']) if CONFIG['target_variable'] in numeric_cols else numeric_cols
    
    for col in numeric_cols_to_impute:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f" - Columna '{col}' imputada con mediana: {median_val:.4f}")
    
    # Eliminar filas con valores faltantes en la variable objetivo
    if df[CONFIG['target_variable']].isnull().any():
        missing_target = df[CONFIG['target_variable']].isnull().sum()
        print(f"\nEliminando {missing_target} filas con valores faltantes en la variable objetivo")
        df = df.dropna(subset=[CONFIG['target_variable']])
    
    missing_after = df.isnull().sum().sum()
    print(f"Valores faltantes después de imputación: {missing_after}")
    
    # 3. Eliminar características altamente correlacionadas
    print("\nIdentificando características altamente correlacionadas...")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > CONFIG['correlation_threshold'])]
        
        if to_drop:
            print(f"Eliminando {len(to_drop)} características con correlación > {CONFIG['correlation_threshold']}:")
            print(to_drop)
            df.drop(columns=to_drop, inplace=True)
        else:
            print(f"No se encontraron características con correlación > {CONFIG['correlation_threshold']}")
    
    # 4. Separación en características (X) y variable objetivo (y)
    print("\nSeparando características y variable objetivo...")
    X = df.drop(columns=[CONFIG['target_variable']])
    y = df[CONFIG['target_variable']]
    
    # Seleccionar solo características numéricas
    feature_names = X.select_dtypes(include=np.number).columns.tolist()
    
    if not feature_names:
        print("\nError: No hay características numéricas para modelar")
        exit()
    
    print(f"Número de características finales: {len(feature_names)}")
    
    # 5. División en conjuntos de entrenamiento y prueba
    print(f"\nDividiendo datos en entrenamiento ({100-CONFIG['TEST_SIZE']*100}%) y prueba ({CONFIG['TEST_SIZE']*100}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X[feature_names], y, 
        test_size=CONFIG['TEST_SIZE'], 
        random_state=CONFIG['SEED'],
        shuffle=True
    )
    
    print(f"Dimensiones de los conjuntos:")
    print(f" - Entrenamiento: {X_train.shape[0]} muestras, {X_train.shape[1]} características")
    print(f" - Prueba: {X_test.shape[0]} muestras")
    
    # 6. Escalado de características (StandardScaler)
    print("\nEscalando características numéricas...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), 
                                columns=feature_names, 
                                index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), 
                               columns=feature_names, 
                               index=X_test.index)
    
    # Verificar que no hay NaNs después del preprocesamiento
    assert X_train_scaled.isnull().sum().sum() == 0, "Hay NaNs en X_train después del preprocesamiento"
    assert X_test_scaled.isnull().sum().sum() == 0, "Hay NaNs en X_test después del preprocesamiento"
    assert y_train.isnull().sum() == 0, "Hay NaNs en y_train después del preprocesamiento"
    assert y_test.isnull().sum() == 0, "Hay NaNs en y_test después del preprocesamiento"
    
    print("\nPreparación de datos completada exitosamente")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names

# Preparar los datos
X_train, X_test, y_train, y_test, feature_names = preparar_datos(df)

# ==============================================================================
# --- 3. Modelado y Evaluación ---
# ==============================================================================
print("\n=== 3. Modelado y Evaluación ===")

def entrenar_evaluar_modelos(X_train, X_test, y_train, y_test, feature_names):
    """Entrena y evalúa diferentes modelos de regresión"""
    print("\n--- Entrenando Modelos ---")
    
    # Diccionario para almacenar resultados
    resultados = {}
    
    # Función auxiliar para evaluar modelos
    def evaluar_modelo(modelo, nombre, X_tr=X_train, X_te=X_test, 
                      y_tr=y_train, y_te=y_test, features=feature_names):
        """Evalúa un modelo y almacena los resultados"""
        # Entrenamiento
        modelo.fit(X_tr, y_tr)
        
        # Predicciones
        y_pred_tr = modelo.predict(X_tr)
        y_pred_te = modelo.predict(X_te)
        
        # Métricas
        mse_tr = mean_squared_error(y_tr, y_pred_tr)
        mse_te = mean_squared_error(y_te, y_pred_te)
        r2_tr = r2_score(y_tr, y_pred_tr)
        r2_te = r2_score(y_te, y_pred_te)
        mae_tr = mean_absolute_error(y_tr, y_pred_tr)
        mae_te = mean_absolute_error(y_te, y_pred_te)
        
        # Almacenar resultados
        resultados[nombre] = {
            'modelo': modelo,
            'MSE_train': mse_tr,
            'MSE_test': mse_te,
            'R2_train': r2_tr,
            'R2_test': r2_te,
            'MAE_train': mae_tr,
            'MAE_test': mae_te
        }
        
        # Mostrar resultados
        print(f"\nResultados para {nombre}:")
        print(f"  MSE (train): {mse_tr:.4f} | MSE (test): {mse_te:.4f}")
        print(f"  R² (train): {r2_tr:.4f} | R² (test): {r2_te:.4f}")
        print(f"  MAE (train): {mae_tr:.4f} | MAE (test): {mae_te:.4f}")
        
        # Coeficientes para modelos lineales
        if hasattr(modelo, 'coef_'):
            print("\n  Coeficientes más importantes (absolutos):")
            coef_df = pd.DataFrame({
                'Variable': features,
                'Coeficiente': modelo.coef_
            }).sort_values(by='Coeficiente', key=abs, ascending=False)
            print(coef_df.head(10).to_string(index=False))
        
        return resultados
    
    # --- 3.1 Regresión Lineal MCO (statsmodels para diagnóstico) ---
    print("\n--- 3.1 Modelo: Regresión Lineal MCO (statsmodels) ---")
    try:
        # Añadir constante para statsmodels
        X_train_sm = sm.add_constant(X_train, has_constant='add')
        X_test_sm = sm.add_constant(X_test, has_constant='add')
        
        # Entrenar modelo OLS
        ols_model = sm.OLS(y_train, X_train_sm).fit()
        print(ols_model.summary(xname=['const'] + feature_names))
        
        # Diagnóstico OLS
        print("\nDiagnóstico del Modelo OLS:")
        print(f"Test Jarque-Bera (Normalidad Residuos): JB={ols_model.jarque_bera[0]:.2f}, p={ols_model.jarque_bera[1]:.3f}")
        print(f"Test Durbin-Watson (Autocorrelación Residuos): DW={ols_model.durbin_watson:.2f}")
        
        # Gráfico de residuos vs ajustados
        y_pred_train_ols = ols_model.predict(X_train_sm)
        residuals_ols = y_train - y_pred_train_ols
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_pred_train_ols, y=residuals_ols)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Valores Ajustados')
        plt.ylabel('Residuos')
        plt.title('Gráfico Residuos vs Ajustados (OLS)')
        plt.grid(True)
        plt.show()
        
        # Q-Q plot de residuos
        plt.figure(figsize=(8, 5))
        sm.qqplot(residuals_ols, line='s')
        plt.title('Q-Q Plot de Residuos (OLS)')
        plt.grid(True)
        plt.show()
        
        # Evaluación en test
        y_pred_test_ols = ols_model.predict(X_test_sm)
        mse_ols = mean_squared_error(y_test, y_pred_test_ols)
        r2_ols = r2_score(y_test, y_pred_test_ols)
        mae_ols = mean_absolute_error(y_test, y_pred_test_ols)
        
        # Almacenar resultados (usando scikit-learn para consistencia)
        lr = LinearRegression()
        lr.fit(X_train, y_train)  # Entrenar nuevamente para almacenar en resultados
        
        resultados['OLS'] = {
            'modelo': lr,
            'MSE_train': mse_ols,
            'MSE_test': mse_ols,
            'R2_train': r2_ols,
            'R2_test': r2_ols,
            'MAE_train': mae_ols,
            'MAE_test': mae_ols
        }
        
    except Exception as e:
        print(f"Error en OLS: {str(e)}")
        resultados['OLS'] = {
            'modelo': None,
            'MSE_train': np.nan,
            'MSE_test': np.nan,
            'R2_train': np.nan,
            'R2_test': np.nan,
            'MAE_train': np.nan,
            'MAE_test': np.nan
        }
    
    # --- 3.2 Método Stepwise (AIC) ---
    print("\n--- 3.2 Modelo: Regresión Stepwise (AIC) ---")
    try:
        # Usar SequentialFeatureSelector para implementar stepwise
        # Seleccionar aproximadamente el 25% de las características como objetivo
        n_features_target = max(1, int(len(feature_names) * 0.25))
        
        sfs = SequentialFeatureSelector(
            LinearRegression(),
            n_features_to_select=n_features_target,
            direction='forward',
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1
        )
        
        print("Ajustando SequentialFeatureSelector (puede tardar)...")
        sfs.fit(X_train, y_train)
        
        # Obtener características seleccionadas
        selected_features = X_train.columns[sfs.get_support()].tolist()
        print(f"\nCaracterísticas seleccionadas por Stepwise ({len(selected_features)}):")
        print(selected_features)
        
        # Entrenar modelo final con características seleccionadas
        lr_stepwise = LinearRegression()
        lr_stepwise.fit(X_train[selected_features], y_train)
        
        # Calcular AIC manualmente
        n = len(y_train)
        k = len(selected_features) + 1  # +1 para el intercepto
        y_pred_train_step = lr_stepwise.predict(X_train[selected_features])
        rss = np.sum((y_train - y_pred_train_step)**2)
        aic_stepwise = n * np.log(rss/n) + 2*k if rss > 0 else np.inf
        
        # Evaluar en test
        y_pred_test_step = lr_stepwise.predict(X_test[selected_features])
        mse_stepwise = mean_squared_error(y_test, y_pred_test_step)
        r2_stepwise = r2_score(y_test, y_pred_test_step)
        mae_stepwise = mean_absolute_error(y_test, y_pred_test_step)
        
        # Almacenar resultados
        resultados['Stepwise'] = {
            'modelo': lr_stepwise,
            'MSE_train': mse_stepwise,
            'MSE_test': mse_stepwise,
            'R2_train': r2_stepwise,
            'R2_test': r2_stepwise,
            'MAE_train': mae_stepwise,
            'MAE_test': mae_stepwise,
            'AIC': aic_stepwise,
            'selected_features': selected_features
        }
        
        print(f"\nEvaluación Stepwise en Test: MSE={mse_stepwise:.4f}, R²={r2_stepwise:.4f}, MAE={mae_stepwise:.4f}, AIC={aic_stepwise:.2f}")
        
    except Exception as e:
        print(f"Error en Stepwise: {str(e)}")
        resultados['Stepwise'] = {
            'modelo': None,
            'MSE_train': np.nan,
            'MSE_test': np.nan,
            'R2_train': np.nan,
            'R2_test': np.nan,
            'MAE_train': np.nan,
            'MAE_test': np.nan,
            'AIC': np.nan,
            'selected_features': None
        }
    
    # --- 3.3 Regresión Ridge (con CV para lambda óptimo) ---
    print("\n--- 3.3 Modelo: Regresión Ridge (con CV) ---")
    try:
        # Definir rango de alphas (lambdas) a probar
        alphas_ridge = np.logspace(-3, 3, 100)  # Desde 10^-3 hasta 10^3
        
        ridge_cv = RidgeCV(
            alphas=alphas_ridge,
            scoring='neg_mean_squared_error',
            cv=5
        )
        
        ridge_cv.fit(X_train, y_train)
        optimal_alpha_ridge = ridge_cv.alpha_
        print(f"Alpha (lambda) óptimo encontrado para Ridge: {optimal_alpha_ridge:.6f}")
        
        # Entrenar modelo final con alpha óptimo
        ridge_final = Ridge(alpha=optimal_alpha_ridge)
        ridge_final.fit(X_train, y_train)
        
        # Evaluar modelo
        evaluar_modelo(ridge_final, "Ridge")
        
        # Almacenar alpha óptimo
        resultados['Ridge']['alpha'] = optimal_alpha_ridge
        
    except Exception as e:
        print(f"Error en Ridge: {str(e)}")
        resultados['Ridge'] = {
            'modelo': None,
            'MSE_train': np.nan,
            'MSE_test': np.nan,
            'R2_train': np.nan,
            'R2_test': np.nan,
            'MAE_train': np.nan,
            'MAE_test': np.nan,
            'alpha': np.nan
        }
    
    # --- 3.4 Regresión Lasso (con CV para lambda óptimo) ---
    print("\n--- 3.4 Modelo: Regresión Lasso (con CV) ---")
    try:
        # Definir rango de alphas (lambdas) a probar
        alphas_lasso = np.logspace(-3, 3, 100)  # Desde 10^-3 hasta 10^3
        
        lasso_cv = LassoCV(
            alphas=alphas_lasso,
            cv=5,
            random_state=CONFIG['SEED'],
            max_iter=10000,
            n_jobs=-1
        )
        
        lasso_cv.fit(X_train, y_train)
        optimal_alpha_lasso = lasso_cv.alpha_
        print(f"Alpha (lambda) óptimo encontrado para Lasso: {optimal_alpha_lasso:.6f}")
        
        # Modelo ya está entrenado con alpha óptimo
        lasso_final = lasso_cv
        
        # Evaluar modelo
        evaluar_modelo(lasso_final, "Lasso")
        
        # Obtener características seleccionadas (coeficientes no cero)
        coefs_lasso = pd.Series(lasso_final.coef_, index=feature_names)
        selected_lasso_features = coefs_lasso[coefs_lasso != 0].index.tolist()
        
        print(f"\nCaracterísticas seleccionadas por Lasso ({len(selected_lasso_features)}):")
        print(selected_lasso_features)
        
        # Almacenar alpha óptimo y características seleccionadas
        resultados['Lasso']['alpha'] = optimal_alpha_lasso
        resultados['Lasso']['selected_features'] = selected_lasso_features
        
    except Exception as e:
        print(f"Error en Lasso: {str(e)}")
        resultados['Lasso'] = {
            'modelo': None,
            'MSE_train': np.nan,
            'MSE_test': np.nan,
            'R2_train': np.nan,
            'R2_test': np.nan,
            'MAE_train': np.nan,
            'MAE_test': np.nan,
            'alpha': np.nan,
            'selected_features': None
        }
    
    # --- 3.5 Regresión Elastic Net (con CV para lambda y alpha óptimos) ---
    print("\n--- 3.5 Modelo: Regresión Elastic Net (con CV) ---")
    try:
        # Definir ratios de mezcla L1/L2 a probar
        l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]  # 1.0 es Lasso puro
        
        elastic_net_cv = ElasticNetCV(
            l1_ratio=l1_ratios,
            alphas=np.logspace(-3, 3, 50),
            cv=5,
            random_state=CONFIG['SEED'],
            max_iter=10000,
            n_jobs=-1
        )
        
        elastic_net_cv.fit(X_train, y_train)
        optimal_alpha_enet = elastic_net_cv.alpha_
        optimal_l1_ratio_enet = elastic_net_cv.l1_ratio_
        
        print(f"Alpha (lambda) óptimo encontrado para ElasticNet: {optimal_alpha_enet:.6f}")
        print(f"L1 Ratio (proporción L1/L2) óptimo encontrado: {optimal_l1_ratio_enet:.2f}")
        
        # Modelo ya está entrenado con parámetros óptimos
        elastic_net_final = elastic_net_cv
        
        # Evaluar modelo
        evaluar_modelo(elastic_net_final, "ElasticNet")
        
        # Obtener características seleccionadas (coeficientes no cero)
        coefs_enet = pd.Series(elastic_net_final.coef_, index=feature_names)
        selected_enet_features = coefs_enet[coefs_enet != 0].index.tolist()
        
        print(f"\nCaracterísticas seleccionadas por ElasticNet ({len(selected_enet_features)}):")
        print(selected_enet_features)
        
        # Almacenar parámetros óptimos y características seleccionadas
        resultados['ElasticNet']['alpha'] = optimal_alpha_enet
        resultados['ElasticNet']['l1_ratio'] = optimal_l1_ratio_enet
        resultados['ElasticNet']['selected_features'] = selected_enet_features
        
    except Exception as e:
        print(f"Error en ElasticNet: {str(e)}")
        resultados['ElasticNet'] = {
            'modelo': None,
            'MSE_train': np.nan,
            'MSE_test': np.nan,
            'R2_train': np.nan,
            'R2_test': np.nan,
            'MAE_train': np.nan,
            'MAE_test': np.nan,
            'alpha': np.nan,
            'l1_ratio': np.nan,
            'selected_features': None
        }
    
    return resultados

# Entrenar y evaluar modelos
resultados = entrenar_evaluar_modelos(X_train, X_test, y_train, y_test, feature_names)

# ==============================================================================
# --- 4. Comparación de Modelos y Selección del Mejor ---
# ==============================================================================
print("\n=== 4. Comparación de Modelos ===")

def comparar_modelos(resultados):
    """Compara los modelos y selecciona el mejor"""
    print("\n--- Comparando Modelos ---")
    
    # Crear DataFrame comparativo
    comparison_data = []
    for model_name, res in resultados.items():
        # Calcular número de características usadas
        n_features = len(feature_names)
        if 'selected_features' in res and res['selected_features'] is not None:
            n_features = len(res['selected_features'])
        elif hasattr(res.get('modelo', None), 'coef_'):
            n_features = (res['modelo'].coef_ != 0).sum()
        
        # Construir string de parámetros
        params_str = ""
        if 'alpha' in res:
            params_str += f"α={res.get('alpha', '--'):.4f}"
        if 'l1_ratio' in res:
            params_str += f", l1={res.get('l1_ratio', '--'):.2f}"
        if 'AIC' in res:
            params_str += f", AIC={res.get('AIC', '--'):.2f}"
        
        comparison_data.append({
            'Modelo': model_name,
            'MSE_test': res.get('MSE_test', np.nan),
            'R2_test': res.get('R2_test', np.nan),
            'MAE_test': res.get('MAE_test', np.nan),
            'N_Features': n_features,
            'Parametros': params_str if params_str else '--'
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Ordenar por R2 (mayor es mejor)
    comparison_df.sort_values(by='R2_test', ascending=False, inplace=True)
    
    print("\nComparación de Modelos (ordenados por R² en test):")
    print(comparison_df.to_string(index=False))
    
    # Seleccionar el mejor modelo según R²
    if not comparison_df.empty and not pd.isna(comparison_df.iloc[0]['R2_test']):
        best_model_name = comparison_df.iloc[0]['Modelo']
        best_model_details = resultados[best_model_name]
        
        print(f"\nMejor modelo: {best_model_name}")
        print(f"  - R² en test: {best_model_details['R2_test']:.4f}")
        print(f"  - MSE en test: {best_model_details['MSE_test']:.4f}")
        print(f"  - MAE en test: {best_model_details['MAE_test']:.4f}")
        
        if 'selected_features' in best_model_details and best_model_details['selected_features']:
            print(f"\nCaracterísticas seleccionadas ({len(best_model_details['selected_features'])}):")
            print(best_model_details['selected_features'])
        elif hasattr(best_model_details.get('modelo', None), 'coef_'):
            coefs = pd.Series(best_model_details['modelo'].coef_, index=feature_names)
            top_features = coefs.abs().sort_values(ascending=False).head(15)
            print("\nTop 15 características más importantes (coeficientes absolutos):")
            print(top_features)
        
        return best_model_name, best_model_details
    else:
        print("\nNo se pudo determinar el mejor modelo (posiblemente todos fallaron)")
        return None, None

# Comparar modelos y seleccionar el mejor
best_model_name, best_model_details = comparar_modelos(resultados)

# ==============================================================================
# --- 5. Conclusiones y Recomendaciones ---
# ==============================================================================
print("\n=== 5. Conclusiones ===")

print("\n--- Resumen del Análisis ---")
print("1. Se realizó un análisis completo del consumo de ClO2 en el proceso de blanqueo de celulosa.")
print("2. Se prepararon los datos adecuadamente, eliminando variables constantes y manejando valores faltantes.")
print("3. Se evaluaron 5 técnicas de modelado diferentes según lo solicitado en el proyecto.")
print(f"4. El mejor modelo seleccionado fue: {best_model_name if best_model_name else 'Ninguno'}")

print("\n--- Recomendaciones ---")
print("1. Implementar el modelo seleccionado para monitorear y predecir el consumo de ClO2.")
print("2. Considerar las variables más importantes identificadas para optimizar el proceso.")
print("3. Validar los resultados con expertos en el proceso de blanqueo.")
print("4. Monitorear periódicamente el desempeño del modelo y recalibrar si es necesario.")

print("\n=== ANÁLISIS COMPLETADO ===")