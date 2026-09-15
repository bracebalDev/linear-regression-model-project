"""
Proyecto: Modelado del Consumo de ClO2 en Proceso de Blanqueo de Celulosa

Descripción:
Este script implementa un análisis completo para predecir el consumo de dióxido de cloro
en el proceso de blanqueo de celulosa, utilizando diferentes técnicas de regresión.
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
from sklearn.impute import SimpleImputer
import warnings

# Configuración inicial
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# ==============================================================================
# --- Configuración Global ---
# ==============================================================================
CONFIG = {
    'ruta_datos': 'Ab19selec.xlsx',       # Archivo de datos en formato Excel
    'hoja_excel': 'Hoja1',                # Nombre de la hoja en el archivo Excel
    'variable_objetivo': 'Consumo Total ClO2',  # Variable objetivo según el problema
    'semilla': 2022,                      # Semilla para reproducibilidad
    'tamano_prueba': 0.20,                # Proporción para conjunto de prueba (80-20)
    'umbral_varianza': 0.01,              # Umbral para eliminar características con baja varianza
    'umbral_correlacion': 0.95            # Umbral para eliminar características altamente correlacionadas
}

# Configurar semilla para reproducibilidad
np.random.seed(CONFIG['semilla'])

# ==============================================================================
# --- 1. Carga y Exploración de Datos ---
# ==============================================================================
def cargar_datos():
    """Carga y realiza exploración inicial del dataset desde archivo Excel"""
    try:
        # Cargar datos desde archivo Excel
        df = pd.read_excel(
            CONFIG['ruta_datos'],
            sheet_name=CONFIG['hoja_excel'],
            parse_dates=[0]             # Intentar parsear la primera columna como fecha
        )
        
        # Eliminar filas completamente vacías si las hay
        df = df.dropna(how='all')
        
        # Forzar conversión de variable objetivo a numérico
        df[CONFIG['variable_objetivo']] = pd.to_numeric(
            df[CONFIG['variable_objetivo']], errors='coerce')
        
        # Eliminar filas donde la variable objetivo no pudo convertirse a numérico
        df = df.dropna(subset=[CONFIG['variable_objetivo']])
        
        print(f"\nDatos cargados exitosamente. Dimensiones: {df.shape}")
        print(f"Columnas disponibles: {list(df.columns)}")
        
        # Verificar si la primera columna es datetime y establecer como índice
        if pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
            df.set_index(df.columns[0], inplace=True)
            print(f"\nColumna '{df.index.name}' establecida como índice (datetime)")
        else:
            print("\nAdvertencia: La primera columna no es datetime - se mantendrá como índice simple")
            df.set_index(df.columns[0], inplace=True)
        
        return df
    
    except FileNotFoundError:
        print(f"\nError: Archivo no encontrado en la ruta: {CONFIG['ruta_datos']}")
        exit()
    except Exception as e:
        print(f"\nError al cargar los datos desde Excel: {str(e)}")
        exit()

def explorar_datos(df):
    """Realiza análisis exploratorio de los datos"""
    print("\n=== Análisis Exploratorio de Datos ===")
    
    # Información general del DataFrame
    print("\nInformación del DataFrame:")
    df.info()
    
    # Estadísticas descriptivas para variables numéricas
    print("\nEstadísticas descriptivas (variables numéricas):")
    columnas_numericas = df.select_dtypes(include=np.number).columns
    print(df[columnas_numericas].describe().transpose())
    
    # Valores faltantes
    print("\nValores faltantes por columna:")
    valores_faltantes = df.isnull().sum()
    print(valores_faltantes[valores_faltantes > 0])
    
    # Columnas con varianza cero (constantes)
    print("\nColumnas con varianza cero (constantes):")
    columnas_constantes = df[columnas_numericas].columns[df[columnas_numericas].var() == 0].tolist()
    print(columnas_constantes if columnas_constantes else "No hay columnas constantes")
    
    # Visualización de la distribución de la variable objetivo
    plt.figure(figsize=(12, 6))
    sns.histplot(df[CONFIG['variable_objetivo']].dropna(), kde=True, bins=30)
    plt.title(f'Distribución de {CONFIG["variable_objetivo"]}')
    plt.xlabel('Consumo Total ClO2 (kg/s)')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Matriz de correlación
    if len(columnas_numericas) > 1:
        plt.figure(figsize=(15, 12))
        matriz_correlacion = df[columnas_numericas].corr()
        sns.heatmap(matriz_correlacion, cmap='coolwarm', center=0, annot=False)
        plt.title('Matriz de Correlación (Variables Numéricas)')
        plt.tight_layout()
        plt.show()
        
        # Correlaciones con la variable objetivo
        correlacion_objetivo = matriz_correlacion[CONFIG['variable_objetivo']].sort_values(
            key=abs, ascending=False)
        print("\nCorrelación con la variable objetivo:")
        print(correlacion_objetivo.head(10))
    
    return df

# ==============================================================================
# --- 2. Preparación de Datos ---
# ==============================================================================
def preparar_datos(df):
    """Preprocesa los datos para modelado"""
    print("\n=== Preparación de Datos ===")
    
    # 1. Eliminar columnas problemáticas
    print("\nEliminando columnas problemáticas...")
    columnas_numericas = df.select_dtypes(include=np.number).columns
    
    # Columnas con varianza cero o muy baja
    columnas_baja_varianza = df[columnas_numericas].columns[
        df[columnas_numericas].var() < CONFIG['umbral_varianza']].tolist()
    
    # Columnas adicionales a eliminar (basado en conocimiento del dominio)
    columnas_a_eliminar = columnas_baja_varianza + [
        col for col in df.columns 
        if any(x in str(col) for x in ['CE05_IP21_547CI1772', 'CE05_IP21_547FI1019'])
    ]
    
    columnas_a_eliminar = list(set(columnas_a_eliminar))  # Eliminar duplicados
    
    if columnas_a_eliminar:
        print(f"Eliminando {len(columnas_a_eliminar)} columnas problemáticas")
        df.drop(columns=columnas_a_eliminar, inplace=True, errors='ignore')
    
    # 2. Manejo de valores faltantes
    print("\nManejando valores faltantes...")
    imputador = SimpleImputer(strategy='median')
    columnas_numericas = df.select_dtypes(include=np.number).columns
    
    # Excluir la variable objetivo de la imputación
    columnas_a_imputar = [col for col in columnas_numericas if col != CONFIG['variable_objetivo']]
    
    df[columnas_a_imputar] = imputador.fit_transform(df[columnas_a_imputar])
    
    # 3. Eliminar características altamente correlacionadas
    print("\nIdentificando características altamente correlacionadas...")
    columnas_numericas = df.select_dtypes(include=np.number).columns
    if len(columnas_numericas) > 1:
        matriz_correlacion = df[columnas_numericas].corr().abs()
        superior = matriz_correlacion.where(np.triu(np.ones(matriz_correlacion.shape), k=1).astype(bool))
        columnas_a_eliminar = [col for col in superior.columns if any(superior[col] > CONFIG['umbral_correlacion'])]
        
        if columnas_a_eliminar:
            print(f"Eliminando {len(columnas_a_eliminar)} características con correlación > {CONFIG['umbral_correlacion']}")
            df.drop(columns=columnas_a_eliminar, inplace=True)
    
    # 4. Separación en características (X) y variable objetivo (y)
    print("\nSeparando características y variable objetivo...")
    X = df.drop(columns=[CONFIG['variable_objetivo']])
    y = df[CONFIG['variable_objetivo']]
    
    # Seleccionar solo características numéricas
    nombres_caracteristicas = X.select_dtypes(include=np.number).columns.tolist()
    
    # 5. División en conjuntos de entrenamiento y prueba
    print(f"\nDividiendo datos en entrenamiento ({100-CONFIG['tamano_prueba']*100}%) y prueba ({CONFIG['tamano_prueba']*100}%)...")
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X[nombres_caracteristicas], y, 
        test_size=CONFIG['tamano_prueba'], 
        random_state=CONFIG['semilla']
    )
    
    # 6. Escalado de características
    print("\nEscalando características numéricas...")
    escalador = StandardScaler()
    X_entrenamiento_escalado = pd.DataFrame(
        escalador.fit_transform(X_entrenamiento), 
        columns=nombres_caracteristicas, 
        index=X_entrenamiento.index
    )
    X_prueba_escalado = pd.DataFrame(
        escalador.transform(X_prueba), 
        columns=nombres_caracteristicas, 
        index=X_prueba.index
    )
    
    print("\nPreparación de datos completada exitosamente")
    return X_entrenamiento_escalado, X_prueba_escalado, y_entrenamiento, y_prueba, nombres_caracteristicas, escalador

# ==============================================================================
# --- 3. Modelado ---
# ==============================================================================
def entrenar_modelo_lineal(X_entrenamiento, y_entrenamiento, X_prueba, y_prueba):
    """Entrena y evalúa un modelo de regresión lineal"""
    modelo = LinearRegression()
    modelo.fit(X_entrenamiento, y_entrenamiento)
    
    predicciones = modelo.predict(X_prueba)
    mse = mean_squared_error(y_prueba, predicciones)
    r2 = r2_score(y_prueba, predicciones)
    
    print("\nResultados Regresión Lineal:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return modelo, mse, r2

def entrenar_ridge(X_entrenamiento, y_entrenamiento, X_prueba, y_prueba):
    """Entrena y evalúa un modelo Ridge con CV para selección de alpha"""
    alphas = np.logspace(-3, 3, 100)
    
    ridge_cv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', cv=5)
    ridge_cv.fit(X_entrenamiento, y_entrenamiento)
    
    print(f"\nMejor alpha para Ridge: {ridge_cv.alpha_:.4f}")
    
    predicciones = ridge_cv.predict(X_prueba)
    mse = mean_squared_error(y_prueba, predicciones)
    r2 = r2_score(y_prueba, predicciones)
    
    print("\nResultados Ridge:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return ridge_cv, mse, r2

def entrenar_lasso(X_entrenamiento, y_entrenamiento, X_prueba, y_prueba):
    """Entrena y evalúa un modelo Lasso con CV para selección de alpha"""
    lasso_cv = LassoCV(
        alphas=np.logspace(-3, 3, 100),
        cv=5,
        random_state=CONFIG['semilla'],
        max_iter=10000
    )
    lasso_cv.fit(X_entrenamiento, y_entrenamiento)
    
    print(f"\nMejor alpha para Lasso: {lasso_cv.alpha_:.4f}")
    
    # Características seleccionadas (coeficientes no cero)
    coef = pd.Series(lasso_cv.coef_, index=X_entrenamiento.columns)
    caracteristicas_seleccionadas = coef[coef != 0].index.tolist()
    print(f"\nCaracterísticas seleccionadas por Lasso ({len(caracteristicas_seleccionadas)}):")
    print(caracteristicas_seleccionadas)
    
    predicciones = lasso_cv.predict(X_prueba)
    mse = mean_squared_error(y_prueba, predicciones)
    r2 = r2_score(y_prueba, predicciones)
    
    print("\nResultados Lasso:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return lasso_cv, mse, r2

def entrenar_elastic_net(X_entrenamiento, y_entrenamiento, X_prueba, y_prueba):
    """Entrena y evalúa un modelo Elastic Net con CV para selección de parámetros"""
    enet_cv = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
        alphas=np.logspace(-3, 3, 50),
        cv=5,
        random_state=CONFIG['semilla'],
        max_iter=10000
    )
    enet_cv.fit(X_entrenamiento, y_entrenamiento)
    
    print(f"\nMejor alpha para ElasticNet: {enet_cv.alpha_:.4f}")
    print(f"Mejor l1_ratio para ElasticNet: {enet_cv.l1_ratio_:.2f}")
    
    # Características seleccionadas (coeficientes no cero)
    coef = pd.Series(enet_cv.coef_, index=X_entrenamiento.columns)
    caracteristicas_seleccionadas = coef[coef != 0].index.tolist()
    print(f"\nCaracterísticas seleccionadas por ElasticNet ({len(caracteristicas_seleccionadas)}):")
    print(caracteristicas_seleccionadas)
    
    predicciones = enet_cv.predict(X_prueba)
    mse = mean_squared_error(y_prueba, predicciones)
    r2 = r2_score(y_prueba, predicciones)
    
    print("\nResultados ElasticNet:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return enet_cv, mse, r2

def entrenar_stepwise(X_entrenamiento, y_entrenamiento, X_prueba, y_prueba):
    """Entrena y evalúa un modelo con selección Stepwise (AIC)"""
    # Usar SequentialFeatureSelector para implementar stepwise
    sfs = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select='auto',
        direction='forward',
        scoring='r2',
        cv=5
    )
    
    print("\nAjustando SequentialFeatureSelector...")
    sfs.fit(X_entrenamiento, y_entrenamiento)
    
    # Obtener características seleccionadas
    caracteristicas_seleccionadas = X_entrenamiento.columns[sfs.get_support()].tolist()
    print(f"\nCaracterísticas seleccionadas por Stepwise ({len(caracteristicas_seleccionadas)}):")
    print(caracteristicas_seleccionadas)
    
    # Entrenar modelo final con características seleccionadas
    modelo = LinearRegression()
    modelo.fit(X_entrenamiento[caracteristicas_seleccionadas], y_entrenamiento)
    
    # Calcular AIC manualmente
    n = len(y_entrenamiento)
    k = len(caracteristicas_seleccionadas) + 1  # +1 para el intercepto
    y_pred = modelo.predict(X_entrenamiento[caracteristicas_seleccionadas])
    rss = np.sum((y_entrenamiento - y_pred)**2)
    aic = n * np.log(rss/n) + 2*k if rss > 0 else np.inf
    
    # Evaluar en test
    predicciones = modelo.predict(X_prueba[caracteristicas_seleccionadas])
    mse = mean_squared_error(y_prueba, predicciones)
    r2 = r2_score(y_prueba, predicciones)
    
    print("\nResultados Stepwise:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  AIC: {aic:.2f}")
    
    return modelo, mse, r2, aic

def diagnosticar_modelo_ols(X_entrenamiento, y_entrenamiento, nombres_caracteristicas):
    """Realiza diagnóstico del modelo OLS usando statsmodels"""
    # Añadir constante
    X_sm = sm.add_constant(X_entrenamiento, has_constant='add')
    
    # Entrenar modelo OLS
    modelo_ols = sm.OLS(y_entrenamiento, X_sm).fit()
    
    # Mostrar resumen
    print("\n=== Diagnóstico del Modelo OLS ===")
    print(modelo_ols.summary(xname=['const'] + nombres_caracteristicas))
    
    # Gráfico de residuos vs ajustados
    plt.figure(figsize=(10, 6))
    sns.residplot(x=modelo_ols.fittedvalues, y=modelo_ols.resid, lowess=True)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Valores Ajustados')
    plt.ylabel('Residuos')
    plt.title('Gráfico Residuos vs Ajustados (OLS)')
    plt.grid(True)
    plt.show()
    
    # Q-Q plot de residuos
    plt.figure(figsize=(8, 5))
    sm.qqplot(modelo_ols.resid, line='s')
    plt.title('Q-Q Plot de Residuos (OLS)')
    plt.grid(True)
    plt.show()
    
    return modelo_ols

# ==============================================================================
# --- 4. Evaluación y Comparación de Modelos ---
# ==============================================================================
def comparar_modelos(resultados):
    """Compara los modelos y selecciona el mejor"""
    print("\n=== Comparación de Modelos ===")
    
    # Crear DataFrame comparativo
    comparacion = pd.DataFrame({
        'Modelo': resultados.keys(),
        'MSE': [res['MSE'] for res in resultados.values()],
        'R²': [res['R²'] for res in resultados.values()],
        'Características': [res.get('Características', len(resultados[modelo]['Coeficientes'])) 
                          for modelo in resultados.keys()]
    })
    
    # Ordenar por R² (mayor es mejor)
    comparacion = comparacion.sort_values('R²', ascending=False)
    
    print("\nResultados comparativos:")
    print(comparacion.to_string(index=False))
    
    # Seleccionar el mejor modelo
    mejor_modelo = comparacion.iloc[0]['Modelo']
    print(f"\nMejor modelo: {mejor_modelo} (R²: {comparacion.iloc[0]['R²']:.4f})")
    
    # Mostrar coeficientes del mejor modelo
    if 'Coeficientes' in resultados[mejor_modelo]:
        coef = pd.Series(resultados[mejor_modelo]['Coeficientes'], 
                        index=resultados[mejor_modelo]['Nombres_Caracteristicas'])
        print("\nCoeficientes más importantes:")
        print(coef.abs().sort_values(ascending=False).head(15))
    
    return mejor_modelo

# ==============================================================================
# --- Función Principal ---
# ==============================================================================
def main():
    print("\n=== INICIO DEL ANÁLISIS ===")
    
    # 1. Carga y exploración de datos
    datos = cargar_datos()
    datos = explorar_datos(datos)
    
    # 2. Preparación de datos
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba, nombres_caracteristicas, escalador = preparar_datos(datos)
    
    # 3. Entrenamiento de modelos
    resultados = {}
    
    # 3.1 Modelo OLS (para diagnóstico)
    modelo_ols = diagnosticar_modelo_ols(X_entrenamiento, y_entrenamiento, nombres_caracteristicas)
    resultados['OLS'] = {
        'MSE': mean_squared_error(y_prueba, modelo_ols.predict(sm.add_constant(X_prueba))),
        'R²': r2_score(y_prueba, modelo_ols.predict(sm.add_constant(X_prueba))),
        'Coeficientes': modelo_ols.params[1:],  # Excluir constante
        'Nombres_Caracteristicas': nombres_caracteristicas
    }
    
    # 3.2 Modelo Lineal
    modelo_lineal, mse_lineal, r2_lineal = entrenar_modelo_lineal(
        X_entrenamiento, y_entrenamiento, X_prueba, y_prueba)
    resultados['Lineal'] = {
        'MSE': mse_lineal,
        'R²': r2_lineal,
        'Coeficientes': modelo_lineal.coef_,
        'Nombres_Caracteristicas': nombres_caracteristicas
    }
    
    # 3.3 Modelo Ridge
    modelo_ridge, mse_ridge, r2_ridge = entrenar_ridge(
        X_entrenamiento, y_entrenamiento, X_prueba, y_prueba)
    resultados['Ridge'] = {
        'MSE': mse_ridge,
        'R²': r2_ridge,
        'Coeficientes': modelo_ridge.coef_,
        'Nombres_Caracteristicas': nombres_caracteristicas
    }
    
    # 3.4 Modelo Lasso
    modelo_lasso, mse_lasso, r2_lasso = entrenar_lasso(
        X_entrenamiento, y_entrenamiento, X_prueba, y_prueba)
    resultados['Lasso'] = {
        'MSE': mse_lasso,
        'R²': r2_lasso,
        'Coeficientes': modelo_lasso.coef_,
        'Nombres_Caracteristicas': nombres_caracteristicas,
        'Características': (modelo_lasso.coef_ != 0).sum()
    }
    
    # 3.5 Modelo Elastic Net
    modelo_enet, mse_enet, r2_enet = entrenar_elastic_net(
        X_entrenamiento, y_entrenamiento, X_prueba, y_prueba)
    resultados['ElasticNet'] = {
        'MSE': mse_enet,
        'R²': r2_enet,
        'Coeficientes': modelo_enet.coef_,
        'Nombres_Caracteristicas': nombres_caracteristicas,
        'Características': (modelo_enet.coef_ != 0).sum()
    }
    
    # 3.6 Modelo Stepwise
    modelo_stepwise, mse_stepwise, r2_stepwise, aic_stepwise = entrenar_stepwise(
        X_entrenamiento, y_entrenamiento, X_prueba, y_prueba)
    resultados['Stepwise'] = {
        'MSE': mse_stepwise,
        'R²': r2_stepwise,
        'AIC': aic_stepwise,
        'Coeficientes': modelo_stepwise.coef_,
        'Nombres_Caracteristicas': X_entrenamiento.columns[SequentialFeatureSelector(
            LinearRegression(), n_features_to_select='auto').fit(X_entrenamiento, y_entrenamiento).get_support()],
        'Características': len(X_entrenamiento.columns[SequentialFeatureSelector(
            LinearRegression(), n_features_to_select='auto').fit(X_entrenamiento, y_entrenamiento).get_support()])
    }
    
    # 4. Comparación de modelos
    mejor_modelo = comparar_modelos(resultados)
    
    print("\n=== ANÁLISIS COMPLETADO ===")

if __name__ == "__main__":
    main()