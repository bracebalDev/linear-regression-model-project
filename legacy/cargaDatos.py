import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- Configuración Inicial ---
# Cambiamos la ruta al archivo DataCopy.xlsx y definimos la variable objetivo
file_path = 'DataCopy.xlsx'  # Asumiendo que está en el mismo directorio
target_variable = 'Consumo Total ClO2'  # Según el archivo Excel, esta es la columna objetivo

# Semilla para reproducibilidad
SEED = 2022
# Proporción para el conjunto de prueba
TEST_SIZE = 0.20

print(f"Semilla aleatoria fijada en: {SEED}")
print(f"Tamaño del conjunto de prueba: {TEST_SIZE*100}%")

# --- 1. Carga de Datos ---
print(f"\n--- 1. Carga de Datos ---")
try:
    # Cargamos el archivo Excel, especificando que la primera columna son fechas
    df = pd.read_excel(file_path, parse_dates=[0])
    print(f"Datos cargados exitosamente desde: {file_path}")
    print(f"Dimensiones del DataFrame: {df.shape}")
    
    # Verificamos y establecemos la columna de fecha como índice
    if pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
        df.set_index(df.columns[0], inplace=True)
        print(f"Columna '{df.index.name}' establecida como índice (tipo datetime).")
    else:
        print("Advertencia: La primera columna no se pudo interpretar como fecha/hora.")
    
    # Mostramos las primeras filas para verificar la carga
    print("\nPrimeras filas del DataFrame:")
    print(df.head())

except FileNotFoundError:
    print(f"Error: El archivo no se encontró en la ruta especificada: {file_path}")
    exit()
except Exception as e:
    print(f"Error al cargar o procesar el archivo Excel: {e}")
    exit()

# --- 2. Comprensión de los Datos (Exploración Inicial) ---
print("\n--- 2. Exploración Inicial de Datos ---")

# Verificar si la variable objetivo existe
if target_variable not in df.columns:
    print(f"Error: La variable objetivo '{target_variable}' no se encontró en las columnas.")
    print("Columnas disponibles:", df.columns.tolist())
    exit()
else:
    print(f"Variable objetivo identificada: '{target_variable}'")

# Mostrar información general del DataFrame
print("\nInformación General del DataFrame (df.info()):")
df.info()

# Mostrar estadísticas descriptivas para variables numéricas
print("\nEstadísticas Descriptivas (df.describe()):")
# Usar pd.options.display para ver más columnas si es necesario
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.describe())

# Identificar valores faltantes
print("\nConteo de Valores Faltantes por Columna:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0]) # Mostrar solo columnas con faltantes

# Identificar columnas con valor constante (varianza cero)
print("\nColumnas con Varianza Cero (Valor Constante):")
constant_columns = df.columns[df.var() == 0]
print(constant_columns.tolist())

# --- Visualizaciones Exploratorias (Ejemplos) ---
print("\nGenerando visualizaciones exploratorias (ejemplos)...")

# Histograma de la variable objetivo
plt.figure(figsize=(10, 6))
sns.histplot(df[target_variable], kde=True)
plt.title(f'Distribución de {target_variable}')
plt.xlabel('Consumo CLO2 (kg/ADt)')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show() # Muestra el gráfico

# Boxplot de la variable objetivo
plt.figure(figsize=(8, 5))
sns.boxplot(y=df[target_variable])
plt.title(f'Boxplot de {target_variable}')
plt.ylabel('Consumo CLO2 (kg/ADt)')
plt.grid(True)
plt.show()

# Serie de tiempo de la variable objetivo (si el índice es datetime)
if isinstance(df.index, pd.DatetimeIndex):
    plt.figure(figsize=(15, 7))
    df[target_variable].plot()
    plt.title(f'Serie de Tiempo de {target_variable}')
    plt.xlabel('Fecha')
    plt.ylabel('Consumo CLO2 (kg/ADt)')
    plt.grid(True)
    plt.show()
else:
    print("Advertencia: No se puede graficar la serie de tiempo porque el índice no es datetime.")

# Mapa de calor de correlaciones (puede ser muy grande con 100 variables)
# Considera seleccionar un subconjunto de variables o usar un umbral
# print("\nCalculando matriz de correlación (puede tardar)...")
# correlation_matrix = df.corr()
# plt.figure(figsize=(20, 15)) # Ajusta el tamaño según necesidad
# sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False) # annot=True es muy lento aquí
# plt.title('Mapa de Calor de Correlaciones')
# plt.show()

print("\n--- Fin de Exploración Inicial ---")
# Nota: El análisis detallado (identificar tipos de variables según PDF, etc.)
# requeriría revisar los nombres y datos específicos de cada columna.
