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
# --- 0. Configuración Inicial ---
# ==============================================================================
# Cambiamos la ruta al archivo Ab19selec.csv y definimos la variable objetivo
file_path = 'Ab19selec.csv'  # Nuevo nombre del archivo
target_variable = 'Consumo Total ClO2'  # Según el archivo Excel, esta es la columna objetivo

# Semilla para reproducibilidad
SEED = 2022
# Proporción para el conjunto de prueba
TEST_SIZE = 0.20

print(f"Semilla aleatoria fijada en: {SEED}")
print(f"Tamaño del conjunto de prueba: {TEST_SIZE*100}%")
print(f"Archivo de datos: {file_path}")
print(f"Variable objetivo: {target_variable}")

# ==============================================================================
# --- 1. Carga de Datos ---
# ==============================================================================
print(f"\n--- 1. Carga de Datos ---")
try:
    # Cargamos el archivo CSV, especificando que la primera columna son fechas (si aplica)
    # Es importante conocer la estructura del CSV. Si la primera columna es la fecha,
    # podemos intentar parsearla. Si no, ajusta el índice según corresponda.
    df = pd.read_csv(file_path, parse_dates=[0])  # Asumimos que la primera columna (índice 0) es la fecha
    print(f"Datos cargados exitosamente desde: {file_path}")
    print(f"Dimensiones iniciales del DataFrame: {df.shape}")

    # Verificamos y establecemos la columna de fecha como índice
    # Ahora verificamos si la primera columna se parseó correctamente como datetime
    if pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
        df.set_index(df.columns[0], inplace=True)
        print(f"Columna '{df.index.name}' establecida como índice (tipo datetime).")
    else:
        print("Advertencia: La primera columna no se pudo interpretar como fecha/hora.")
        # Si la fecha está en otra columna o no se parseó, podrías intentar lo siguiente:
        # 1. Identificar el nombre correcto de la columna de fecha en el CSV.
        # 2. Usar parse_dates=['NombreDeLaColumna'] en pd.read_csv().
        # 3. Establecer el índice usando df.set_index('NombreDeLaColumna', inplace=True).
        # Ejemplo (si la columna de fecha se llama 'Fecha'):
        # df = pd.read_csv(file_path, parse_dates=['Fecha'])
        # df.set_index('Fecha', inplace=True)
        # print("Columna 'Fecha' establecida como índice.")

    # Mostramos las primeras filas para verificar la carga
    print("\nPrimeras filas del DataFrame (head):")
    print(df.head())

except FileNotFoundError:
    print(f"Error CRÍTICO: El archivo no se encontró en la ruta especificada: {file_path}")
    exit()
except Exception as e:
    print(f"Error CRÍTICO al cargar o procesar el archivo CSV: {e}")
    exit()

# ==============================================================================
# --- 2. Comprensión de los Datos (Exploración Inicial) ---
# ==============================================================================
print("\n--- 2. Exploración Inicial de Datos ---")

# Verificar si la variable objetivo existe
if target_variable not in df.columns:
    print(f"Error CRÍTICO: La variable objetivo '{target_variable}' no se encontró en las columnas.")
    print("Columnas disponibles:", df.columns.tolist())
    exit()
else:
    print(f"Variable objetivo '{target_variable}' encontrada.")

# Mostrar información general del DataFrame
print("\nInformación General del DataFrame (df.info()):")
# Aumentar max_cols para ver todos los tipos de datos si son muchas columnas
with pd.option_context('display.max_info_columns', df.shape[1] + 1):
    df.info()


# Mostrar estadísticas descriptivas para variables numéricas
print("\nEstadísticas Descriptivas (df.describe()):")
# Usar pd.options.display para ver más columnas si es necesario
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.describe())

# Identificar valores faltantes
print("\nConteo de Valores Faltantes por Columna (solo si hay):")
missing_values = df.isnull().sum()
missing_values_filtered = missing_values[missing_values > 0]
if not missing_values_filtered.empty:
    print(missing_values_filtered)
else:
    print("No se encontraron valores faltantes.")


# Identificar columnas con valor constante (varianza cero) - se hará en preparación
# print("\nColumnas con Varianza Cero (Valor Constante):")
# numeric_cols_var = df.select_dtypes(include=np.number).columns
# variances_check = df[numeric_cols_var].var()
# constant_columns_check = variances_check[variances_check == 0].index.tolist()
# print(constant_columns_check)

# --- Visualizaciones Exploratorias (Ejemplos) ---
print("\nGenerando visualizaciones exploratorias (ejemplos)...")

try:
    # Histograma de la variable objetivo
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target_variable], kde=True, bins=30) # Ajustar bins si es necesario
    plt.title(f'Distribución de {target_variable}')
    plt.xlabel('Consumo CLO2 (kg/ADt)')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Boxplot de la variable objetivo
    plt.figure(figsize=(8, 5))
    sns.boxplot(y=df[target_variable])
    plt.title(f'Boxplot de {target_variable}')
    plt.ylabel('Consumo CLO2 (kg/ADt)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Serie de tiempo de la variable objetivo (si el índice es datetime)
    if isinstance(df.index, pd.DatetimeIndex):
        plt.figure(figsize=(15, 7))
        df[target_variable].plot()
        plt.title(f'Serie de Tiempo de {target_variable}')
        plt.xlabel('Fecha')
        plt.ylabel('Consumo CLO2 (kg/ADt)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("Advertencia: No se puede graficar la serie de tiempo porque el índice no es datetime.")

except Exception as e:
    print(f"Error durante la generación de gráficos exploratorios: {e}")


# Mapa de calor de correlaciones (Opcional, puede ser muy grande y lento)
# print("\nCalculando matriz de correlación (puede tardar)...")
# try:
#     # Seleccionar solo columnas numéricas para corr()
#     numeric_df_corr = df.select_dtypes(include=np.number)
#     if not numeric_df_corr.empty:
#         correlation_matrix = numeric_df_corr.corr()
#         plt.figure(figsize=(20, 15)) # Ajusta el tamaño según necesidad
#         sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False) # annot=True es muy lento aquí
#         plt.title('Mapa de Calor de Correlaciones (Numéricas)')
#         plt.tight_layout()
#         plt.show()
#     else:
#         print("No hay suficientes columnas numéricas para calcular la correlación.")
# except Exception as e:
#     print(f"Error calculando o mostrando mapa de calor de correlación: {e}")


print("\n--- Fin de Exploración Inicial ---")


# ==============================================================================
# --- 3. Preparación de Datos ---
# ==============================================================================
print("\n--- 3. Preparación de Datos ---")

# --- Limpieza ---

# 3.1 Eliminar columnas con varianza cero o constantes identificadas
print("\nEliminando columnas con varianza cero o constantes...")
# Primero verificamos columnas numéricas con varianza cero
numeric_cols_var = df.select_dtypes(include=np.number).columns
if not numeric_cols_var.empty:
    variances = df[numeric_cols_var].var()
    constant_columns_auto = variances[variances == 0].index.tolist()
else:
    constant_columns_auto = []
    print("Advertencia: No se encontraron columnas numéricas para calcular varianza.")


# Columnas adicionales a eliminar basadas en el análisis previo del dataset
additional_columns_to_drop = [
    'CE05_IP21_547AI1011',  # Kappa Entr Prensa preblanqueo L1 - valor constante en las primeras filas
    'CE05_IP21_547AI1137',  # PH etapa D0 - casi constante
    'CE05_IP21_547AI1250',  # Minikappa EOp (DCS) - valor constante 1.4
    'CE05_IP21_547CI1772',  # Conductividad Prensa PreBlanqueo - todos 0
    'CE05_IP21_547FI1019',  # CE05_IP21_547FI1019 - todos 0
]

# Combinamos las columnas a eliminar, asegurándonos que existan en el DataFrame
all_columns_to_drop_potential = list(set(constant_columns_auto + additional_columns_to_drop))
all_columns_to_drop = [col for col in all_columns_to_drop_potential if col in df.columns]

if all_columns_to_drop:
    print(f"Eliminando {len(all_columns_to_drop)} columnas: {all_columns_to_drop}")
    df.drop(columns=all_columns_to_drop, inplace=True)
    print(f"Dimensiones después de eliminar constantes: {df.shape}")
else:
    print("No se eliminaron columnas por varianza cero o constantes predefinidas.")

# 3.2 Manejar valores faltantes
print("\nManejando valores faltantes...")
missing_before = df.isnull().sum().sum()
print(f"Total de valores faltantes antes del manejo: {missing_before}")

if missing_before > 0:
    # Estrategia: Imputar con la mediana para columnas numéricas (excepto target)
    numeric_cols = df.select_dtypes(include=np.number).columns
    if target_variable in numeric_cols:
        # Asegurarse de no imputar la variable objetivo
        numeric_cols_to_impute = numeric_cols.drop(target_variable)
    else:
        numeric_cols_to_impute = numeric_cols

    print(f"Imputando valores faltantes en {len(numeric_cols_to_impute)} columnas numéricas con la mediana...")
    imputed_cols_count = 0
    for col in numeric_cols_to_impute:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            imputed_cols_count += 1
            # print(f" - Columna '{col}' imputada con mediana: {median_val:.4f}") # Descomentar para detalle
    print(f"Se imputaron {imputed_cols_count} columnas.")

    # Verificar si aún quedan faltantes (podrían ser no numéricos o en target)
    missing_after = df.isnull().sum().sum()
    print(f"Total de valores faltantes después de la imputación: {missing_after}")
    if missing_after > 0:
        print("Advertencia: Aún quedan valores faltantes.")
        print(df.isnull().sum()[df.isnull().sum() > 0])
        # Considera eliminar filas con faltantes en la variable objetivo si son pocas y es apropiado
        print(f"Considerando eliminar filas con NaNs restantes en la variable objetivo '{target_variable}'...")
        initial_rows = df.shape[0]
        df.dropna(subset=[target_variable], inplace=True)
        rows_after_drop = df.shape[0]
        print(f"Se eliminaron {initial_rows - rows_after_drop} filas con NaNs en '{target_variable}'.")
        print(f"Dimensiones después de eliminar NaNs en target: {df.shape}")

        # Re-verificar faltantes
        missing_final = df.isnull().sum().sum()
        if missing_final > 0:
            print("¡Advertencia! Aún quedan valores faltantes después de tratar la variable objetivo. Revisar columnas no numéricas.")
            print(df.isnull().sum()[df.isnull().sum() > 0])
        else:
            print("Todos los valores faltantes han sido tratados.")
else:
    print("No había valores faltantes que manejar.")


# --- Separación de Datos (Features y Target) ---
print("\nSeparando características (X) y variable objetivo (y)...")
# Asegurarse de que la variable objetivo todavía existe después de la limpieza
if target_variable not in df.columns:
    print(f"Error CRÍTICO: La variable objetivo '{target_variable}' fue eliminada durante la limpieza.")
    exit()

X = df.drop(columns=[target_variable])
y = df[target_variable]

# Asegurarse de que X solo contenga columnas numéricas para los modelos estándar
X = X.select_dtypes(include=np.number)
print(f"Se seleccionaron {X.shape[1]} características numéricas para X.")


print(f"Dimensiones de X (características numéricas): {X.shape}")
print(f"Dimensiones de y (objetivo): {y.shape}")

# Guardar los nombres de las características para usarlos después
feature_names = X.columns.tolist()

# --- División en Entrenamiento y Prueba ---
print(f"\nDividiendo los datos en entrenamiento ({100-TEST_SIZE*100:.0f}%) y prueba ({TEST_SIZE*100:.0f}%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=SEED,
    shuffle=True # shuffle=True es común para modelos de regresión estándar
)

print(f"Tamaño X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Tamaño X_test: {X_test.shape}, y_test: {y_test.shape}")

# --- Transformación (Escalado) ---
# Escalar todas las características en X_train/X_test ya que seleccionamos solo numéricas antes
if not X_train.empty:
    print(f"\nEscalando {X_train.shape[1]} características numéricas usando StandardScaler...")
    scaler = StandardScaler()

    # Ajustar el scaler SOLO con los datos de entrenamiento
    # Usar .values para obtener array numpy y reasignar a DataFrame para mantener nombres
    X_train_scaled_np = scaler.fit_transform(X_train)
    X_train_proc = pd.DataFrame(X_train_scaled_np, index=X_train.index, columns=feature_names)


    # Aplicar la misma transformación a los datos de prueba
    X_test_scaled_np = scaler.transform(X_test)
    X_test_proc = pd.DataFrame(X_test_scaled_np, index=X_test.index, columns=feature_names)


    print("Escalado completado.")

    # Verificar que no haya NaNs introducidos por el escalado (no debería pasar)
    if X_train_proc.isnull().sum().sum() > 0 or X_test_proc.isnull().sum().sum() > 0:
        print("¡Advertencia! Se encontraron NaNs después del escalado. Revisa los pasos.")
    else:
        print("Verificación post-escalado: No se encontraron NaNs.")

else:
    print("\nNo hay características numéricas para escalar (X_train está vacío).")
    # Usar los dataframes originales si no hubo escalado (aunque vacíos)
    X_train_proc = X_train
    X_test_proc = X_test


print("\n--- Fin de Preparación de Datos ---")
# Ahora X_train_proc, X_test_proc, y_train, y_test están listos para el modelado


# ==============================================================================
# --- 4. Modelado y Evaluación ---
# ==============================================================================
print("\n--- 4. Modelado y Evaluación ---")

# Diccionario para almacenar resultados de los modelos
results = {}

# Verificar si hay datos para modelar
if X_train_proc.empty or y_train.empty:
    print("Error CRÍTICO: No hay datos de entrenamiento disponibles para el modelado.")
    exit()


# --- 4.1 Regresión Lineal MCO (statsmodels para diagnóstico) ---
print("\n--- 4.1 Modelo: Regresión Lineal MCO (statsmodels) ---")
# Añadir constante (intercepto) para statsmodels
# Asegurarse de que los índices coincidan si se usa pd.DataFrame
X_train_sm = sm.add_constant(X_train_proc, has_constant='add')
X_test_sm = sm.add_constant(X_test_proc, has_constant='add')


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
    plt.tight_layout()
    plt.show()

    # 2. Normalidad de Residuos: Q-Q plot
    plt.figure(figsize=(8, 5))
    sm.qqplot(residuals_ols, line='s')
    plt.title('Q-Q Plot de Residuos (OLS)')
    plt.grid(True)
    plt.tight_layout()
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
    # Usamos LinearRegression como estimador base
    estimator = LinearRegression()

    # Configurar SFS para selección forward
    # n_features_to_select='auto' requiere scoring que devuelva un score único,
    # y tol para detenerse. O especificar un número fijo/float.
    # Usaremos 'best' con k_features si usamos mlxtend, o un número/float aquí.
    # Probemos seleccionando la mitad de las características como punto de partida,
    # o un número más pequeño si se sospecha que pocas son relevantes.
    n_total_features = X_train_proc.shape[1]
    n_features_target = max(1, int(n_total_features * 0.25)) # Ejemplo: seleccionar hasta 25% de features
    print(f"Stepwise: Intentando seleccionar hasta {n_features_target} características.")

    sfs = SequentialFeatureSelector(
        estimator,
        n_features_to_select=n_features_target, # Número objetivo de características
        direction='forward',                      # Stepwise forward
        scoring='neg_mean_squared_error', # Métrica para evaluar subsets
        cv=5,                                     # Validación cruzada
        n_jobs=-1                                 # Usar todos los cores disponibles
    )

    print("Ajustando SequentialFeatureSelector (puede tardar)...")
    sfs.fit(X_train_proc, y_train)
    print("Ajuste de SFS completado.")

    # Obtener características seleccionadas
    selected_features_mask = sfs.get_support()
    selected_features = X_train_proc.columns[selected_features_mask].tolist()

    if not selected_features:
        print("Advertencia: Stepwise no seleccionó ninguna característica.")
        results['Stepwise'] = {'MSE': np.nan, 'R2': np.nan, 'MAE': np.nan, 'AIC': np.nan, 'Model': None, 'Features': []}
    else:
        print(f"Características seleccionadas por Stepwise ({len(selected_features)}): {selected_features}")

        # Entrenar modelo final con características seleccionadas
        lr_stepwise = LinearRegression()
        lr_stepwise.fit(X_train_proc[selected_features], y_train)

        # Calcular AIC manualmente (para comparación con otros modelos)
        n = len(y_train)
        k = len(selected_features) + 1   # +1 para el intercepto
        y_pred_train_step = lr_stepwise.predict(X_train_proc[selected_features])
        rss = np.sum((y_train - y_pred_train_step)**2)
        # Evitar log(0) o división por cero si rss es muy pequeño o n es 0
        if rss <= 0 or n == 0:
            aic_stepwise = np.inf # O manejar como NaN
        else:
            aic_stepwise = n * np.log(rss/n) + 2*k


        # Evaluación en Test
        # Asegurarse que X_test_proc tenga las columnas seleccionadas
        X_test_stepwise = X_test_proc[selected_features]
        y_pred_test_step = lr_stepwise.predict(X_test_stepwise)
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
try:
    alphas_ridge = np.logspace(-6, 6, 13)  # 1e-6 a 1e6
    # Usar scoring='neg_mean_squared_error' o 'r2'
    ridge_cv = RidgeCV(alphas=alphas_ridge, scoring='neg_mean_squared_error', cv=5)
    ridge_cv.fit(X_train_proc, y_train)

    optimal_alpha_ridge = ridge_cv.alpha_
    print(f"Alpha (lambda) óptimo encontrado para Ridge: {optimal_alpha_ridge:.6f}")

    # El modelo ridge_cv ya está entrenado con el mejor alpha
    ridge_final = ridge_cv # O re-entrenar: Ridge(alpha=optimal_alpha_ridge).fit(X_train_proc, y_train)

    # Evaluación en Test
    y_pred_test_ridge = ridge_final.predict(X_test_proc)
    mse_ridge = mean_squared_error(y_test, y_pred_test_ridge)
    r2_ridge = r2_score(y_test, y_pred_test_ridge)
    mae_ridge = mean_absolute_error(y_test, y_pred_test_ridge)
    results['Ridge'] = {'MSE': mse_ridge, 'R2': r2_ridge, 'MAE': mae_ridge, 'Model': ridge_final, 'Alpha': optimal_alpha_ridge}
    print(f"Evaluación Ridge (alpha={optimal_alpha_ridge:.6f}) en Test: MSE={mse_ridge:.4f}, R2={r2_ridge:.4f}, MAE={mae_ridge:.4f}")

except Exception as e:
    print(f"Error durante el ajuste de RidgeCV: {e}")
    results['Ridge'] = {'MSE': np.nan, 'R2': np.nan, 'MAE': np.nan, 'Model': None, 'Alpha': np.nan}


# --- 4.4 Regresión Lasso (con CV para lambda óptimo) ---
print("\n--- 4.4 Modelo: Regresión Lasso (con CV) ---")
try:
    # LassoCV automáticamente busca el mejor alpha (lambda)
    lasso_cv = LassoCV(cv=5, random_state=SEED, n_jobs=-1, max_iter=10000) # Aumentar max_iter si no converge
    lasso_cv.fit(X_train_proc, y_train)

    optimal_alpha_lasso = lasso_cv.alpha_
    print(f"Alpha (lambda) óptimo encontrado para Lasso: {optimal_alpha_lasso:.6f}")

    # El modelo ajustado en lasso_cv ya usa el alpha óptimo
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

except Exception as e:
    print(f"Error durante el ajuste de LassoCV: {e}")
    results['Lasso'] = {'MSE': np.nan, 'R2': np.nan, 'MAE': np.nan, 'Model': None, 'Alpha': np.nan, 'Features': []}


# --- 4.5 Regresión Elastic Net (con CV para lambda y alpha óptimos) ---
print("\n--- 4.5 Modelo: Regresión Elastic Net (con CV) ---")
try:
    # ElasticNetCV busca el mejor alpha (lambda total) y l1_ratio (alpha en la fórmula del PDF)
    l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0] # Rango de mezcla entre L1 y L2

    elastic_net_cv = ElasticNetCV(
        l1_ratio=l1_ratios,
        cv=5,
        random_state=SEED,
        n_jobs=-1,
        max_iter=10000 # Aumentar si no converge
    )
    elastic_net_cv.fit(X_train_proc, y_train)

    optimal_alpha_enet = elastic_net_cv.alpha_ # Lambda total
    optimal_l1_ratio_enet = elastic_net_cv.l1_ratio_ # Alpha (mezcla)
    print(f"Alpha (lambda) óptimo encontrado para ElasticNet: {optimal_alpha_enet:.6f}")
    print(f"L1 Ratio (alpha mezcla) óptimo encontrado para ElasticNet: {optimal_l1_ratio_enet:.2f}")

    # El modelo ajustado ya usa los parámetros óptimos
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

except Exception as e:
    print(f"Error durante el ajuste de ElasticNetCV: {e}")
    results['ElasticNet'] = {'MSE': np.nan, 'R2': np.nan, 'MAE': np.nan, 'Model': None, 'Alpha': np.nan, 'L1_Ratio': np.nan, 'Features': []}


# --- 5. Comparación de Modelos y Selección ---
print("\n--- 5. Comparación de Resultados en Conjunto de Prueba ---")

# Crear DataFrame comparativo
# Usar .get con default NaN para manejar casos donde un modelo falló
comparison_data = []
for model_name, res in results.items():
    # Calcular número de features usadas
    n_features = np.nan
    if 'Features' in res and res['Features'] is not None: # Stepwise, Lasso, ENet
        n_features = len(res['Features'])
    elif 'Model' in res and res['Model'] is not None:
        if hasattr(res['Model'], 'coef_'): # LassoCV, ElasticNetCV, RidgeCV models
            # For Ridge, all features are technically used (non-zero coefs)
            if 'Ridge' in model_name:
                n_features = len(feature_names) # Or len(res['Model'].coef_)
            else: # OLS, Lasso, ENet coefs
                try:
                    n_features = (res['Model'].coef_ != 0).sum()
                except AttributeError: # Handle statsmodels OLSResults
                    if hasattr(res['Model'], 'params'):
                        # Exclude const for feature count
                        n_features = len(res['Model'].params) -1
        elif hasattr(res['Model'], 'params'): # statsmodels OLSResults
            # Exclude const for feature count
            n_features = len(res['Model'].params) -1


    # Construir parámetros string
    params_str = ""
    if 'Alpha' in res:
        params_str += f"α={res.get('Alpha', '--'):.4f}"
    if 'L1_Ratio' in res:
        params_str += f", l1={res.get('L1_Ratio', '--'):.2f}"

    comparison_data.append({
        'Modelo': model_name,
        'MSE': res.get('MSE', np.nan),
        'R2': res.get('R2', np.nan),
        'MAE': res.get('MAE', np.nan),
        'AIC': res.get('AIC', np.nan), # Solo Stepwise lo calcula aquí
        'N_Features': n_features,
        'Parametros': params_str if params_str else '--'
    })

comparison_df = pd.DataFrame(comparison_data)

# Ordenar por R2 (mayor es mejor), manejar NaNs
comparison_df.sort_values(by='R2', ascending=False, inplace=True, na_position='last')

print("\nComparación de Modelos (ordenados por R2 descendente):")
# Usar formato para mejor lectura
pd.options.display.float_format = '{:.4f}'.format
print(comparison_df.to_string(index=False))

# --- Selección del Mejor Modelo ---
if not comparison_df.empty and not pd.isna(comparison_df.iloc[0]['R2']):
    best_model_row = comparison_df.iloc[0]
    best_model_name = best_model_row['Modelo']
    best_model_details = results[best_model_name]

    print(f"\nModelo con mejor rendimiento (según R2): {best_model_name}")
    print(f"  - R2: {best_model_row['R2']:.4f}")
    print(f"  - MSE: {best_model_row['MSE']:.4f}")
    print(f"  - MAE: {best_model_row['MAE']:.4f}")
    if not pd.isna(best_model_row['AIC']):
        print(f"  - AIC: {best_model_row['AIC']:.2f}")
    print(f"  - N_Features: {best_model_row['N_Features']:.0f}")
    print(f"  - Parametros: {best_model_row['Parametros']}")


    # Mostrar variables importantes del mejor modelo
    print(f"\nVariables más importantes para el modelo {best_model_name}:")
    best_model_obj = best_model_details.get('Model')
    best_features = best_model_details.get('Features') # Para Lasso, ENet, Stepwise

    if best_features is not None: # Si el modelo hizo selección explícita
        print(f"  Características seleccionadas ({len(best_features)}): {best_features}")
        # Si queremos ver los coeficientes de estas features:
        if hasattr(best_model_obj, 'coef_') and hasattr(best_model_obj, 'intercept_'):
            try:
                # Necesitamos mapear coeficientes a nombres de features seleccionadas
                if 'Stepwise' in best_model_name:
                    # Modelo fue entrenado solo con features seleccionadas
                    coefs = pd.Series(best_model_obj.coef_, index=best_features)
                else: # Lasso, ENet (modelo tiene todos los coefs, algunos son 0)
                    all_coefs = pd.Series(best_model_obj.coef_, index=feature_names)
                    coefs = all_coefs[best_features] # Filtrar solo los no-cero

                print("\n  Coeficientes de las características seleccionadas (Top 15 Abs):")
                print(coefs.abs().sort_values(ascending=False).head(15))
                # print(coefs.sort_values(key=abs, ascending=False).head(15)) # Con signo

            except Exception as e:
                print(f"  No se pudieron mostrar coeficientes para {best_model_name}: {e}")

    elif best_model_obj is not None: # Para OLS, Ridge (usan todas las features)
        if hasattr(best_model_obj, 'coef_'): # Modelos scikit-learn (Ridge)
            coefs = pd.Series(best_model_obj.coef_, index=feature_names)
            print("  Top 15 variables (coeficientes absolutos):")
            print(coefs.abs().sort_values(ascending=False).head(15))
        elif hasattr(best_model_obj, 'params'): # Modelo statsmodels (OLS)
            # Excluir constante
            coefs = best_model_obj.params.drop('const', errors='ignore')
            print("  Top 15 variables (coeficientes absolutos):")
            print(coefs.abs().sort_values(ascending=False).head(15))
        else:
            print("  No se pudieron extraer coeficientes para este modelo.")
    else:
        print("  No hay información de modelo o características disponible.")

else:
    print("\nNo se pudo determinar el mejor modelo (posiblemente todos fallaron o no hay resultados).")


print("\n--- Fin del Análisis ---")

# --- Conclusiones Preliminares (a refinar con los resultados específicos) ---
# 1. Revisar la tabla de comparación para identificar el modelo con mejor balance R2/MAE/MSE y N_Features.
# 2. Analizar los coeficientes/variables seleccionadas del mejor modelo para entender qué factores influyen más en 'Consumo Total ClO2'.
# 3. Evaluar los diagnósticos del modelo OLS (linealidad, normalidad de residuos, etc.) si este es candidato.
# 4. Considerar la interpretabilidad vs. precisión. Un modelo como Lasso/ElasticNet puede ser preferible si reduce significativamente la complejidad (N_Features) sin perder mucha precisión.
# 5. Validar los hallazgos con expertos del proceso de blanqueo.