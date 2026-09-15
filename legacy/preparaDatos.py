# Continuación del script anterior... Asume que 'df' ya está cargado

print("\n--- 3. Preparación de Datos ---")

# --- Limpieza ---

# 3.1 Eliminar columnas con varianza cero (valor constante)
# Primero verificamos columnas numéricas
numeric_cols = df.select_dtypes(include=np.number).columns
variances = df[numeric_cols].var()
constant_columns = variances[variances == 0].index.tolist()

# Columnas adicionales a eliminar basadas en el análisis del dataset
# Identificamos columnas con valores constantes o sin variación
additional_columns_to_drop = [
    'CE05_IP21_547AI1011',  # Kappa Entr Prensa preblanqueo L1 - valor constante en las primeras filas
    'CE05_IP21_547AI1137',  # PH etapa D0 - casi constante
    'CE05_IP21_547AI1250',  # Minikappa EOp (DCS) - valor constante 1.4
    'CE05_IP21_547CI1772',  # Conductividad Prensa PreBlanqueo - todos 0
    'CE05_IP21_547FI1019',  # CE05_IP21_547FI1019 - todos 0
]

# Combinamos las columnas a eliminar
all_columns_to_drop = list(set(constant_columns + additional_columns_to_drop))

if all_columns_to_drop:
    print(f"Eliminando {len(all_columns_to_drop)} columnas con varianza cero o valores constantes: {all_columns_to_drop}")
    df.drop(columns=all_columns_to_drop, inplace=True)
    print(f"Dimensiones después de eliminar constantes: {df.shape}")
else:
    print("No se encontraron columnas con varianza cero para eliminar.")

# 3.2 Manejar valores faltantes
print("\nManejando valores faltantes...")
missing_before = df.isnull().sum().sum()
print(f"Total de valores faltantes antes del manejo: {missing_before}")

# Estrategia: Imputar con la mediana para columnas numéricas
numeric_cols = df.select_dtypes(include=np.number).columns
if target_variable in numeric_cols:
    numeric_cols_to_impute = numeric_cols.drop(target_variable)
else:
    numeric_cols_to_impute = numeric_cols

print(f"Imputando valores faltantes en {len(numeric_cols_to_impute)} columnas numéricas con la mediana...")
for col in numeric_cols_to_impute:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f" - Columna '{col}' imputada con mediana: {median_val:.4f}")

# Verificar si aún quedan faltantes
missing_after = df.isnull().sum().sum()
print(f"Total de valores faltantes después de la imputación: {missing_after}")
if missing_after > 0:
    print("Advertencia: Aún quedan valores faltantes. Revisar columnas no numéricas o la variable objetivo.")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    # Resto del código permanece igual...
    # Considera eliminar filas con faltantes en la variable objetivo si son pocas
    # df.dropna(subset=[target_variable], inplace=True)


# --- Separación de Datos (Features y Target) ---
print("\nSeparando características (X) y variable objetivo (y)...")
X = df.drop(columns=[target_variable])
y = df[target_variable]

print(f"Dimensiones de X (características): {X.shape}")
print(f"Dimensiones de y (objetivo): {y.shape}")

# Guardar los nombres de las características para usarlos después
feature_names = X.columns.tolist()

# --- División en Entrenamiento y Prueba ---
print(f"\nDividiendo los datos en entrenamiento ({100-TEST_SIZE*100}%) y prueba ({TEST_SIZE*100}%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=SEED,
    # shuffle=False # Considera shuffle=False si el orden temporal es crucial y no usas modelos de series de tiempo
    shuffle=True # shuffle=True es común para modelos de regresión estándar
)

print(f"Tamaño X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Tamaño X_test: {X_test.shape}, y_test: {y_test.shape}")

# --- Transformación (Escalado) ---
# Solo escalar características numéricas
# Identificar columnas numéricas en X_train
numeric_features = X_train.select_dtypes(include=np.number).columns

if len(numeric_features) > 0:
    print(f"\nEscalando {len(numeric_features)} características numéricas usando StandardScaler...")
    scaler = StandardScaler()

    # Ajustar el scaler SOLO con los datos de entrenamiento
    X_train_scaled = X_train.copy() # Evitar SettingWithCopyWarning
    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])

    # Aplicar la misma transformación a los datos de prueba
    X_test_scaled = X_test.copy()
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

    print("Escalado completado.")
    # Guardar los dataframes escalados para el modelado
    X_train_proc = X_train_scaled
    X_test_proc = X_test_scaled
else:
    print("\nNo se encontraron características numéricas para escalar.")
    # Usar los dataframes originales si no hubo escalado
    X_train_proc = X_train
    X_test_proc = X_test


# Verificar que no haya NaNs introducidos por el escalado (no debería pasar con la imputación previa)
if X_train_proc.isnull().sum().sum() > 0 or X_test_proc.isnull().sum().sum() > 0:
     print("¡Advertencia! Se encontraron NaNs después del preprocesamiento. Revisa los pasos.")

print("\n--- Fin de Preparación de Datos ---")
# Ahora X_train_proc, X_test_proc, y_train, y_test están listos para el modelado
