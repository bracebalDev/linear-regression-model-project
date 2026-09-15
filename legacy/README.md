# 📁 Archivo Histórico - Versión Inicial (2024)

Este directorio contiene los scripts y notas del desarrollo preliminar realizado en **2024**.

## 📌 Contexto y Diagnóstico del Código Original

En la versión inicial de 2024, el equipo implementó los primeros scripts para abordar el problema de modelado de regresión en la industria de la celulosa:

### Aspectos que estuvieron bien:
1. **Identificación preliminar del problema**: Se reconoció la importancia de predecir el consumo de dióxido de cloro ($ClO_2$) y su impacto económico.
2. **Uso de librerías estándar**: Incorporación de `scikit-learn`, `statsmodels`, `pandas` y `seaborn`.
3. **Fijación de semilla**: Definición de la semilla `SEED = 2022` y división 80/20.

### Oportunidades de mejora identificadas (Qué estuvo mal o incompleto):
1. **Manejo de grandes volúmenes de datos**:
   - Debido a tiempos de procesamiento en Excel, se trabajó provisionalmente con una muestra recortada (`DataCopy.xlsx` de solo 9 filas) en lugar del dataset completo (`Ab19selec.xlsx` con 20,128 filas).
2. **Estructura monolítica y scripts acoplados**:
   - `preparaDatos.py` y `modelaEvalua.py` dependían de variables en memoria definidas en otros scripts o en ejecuciones secuenciales sin modularización (`X_train_proc`, etc.).
3. **Data Leakage y Tautología en Variables**:
   - En algunas ejecuciones se incluyeron sumandos directos del consumo total dentro de las variables predictoras, enmascarando el comportamiento real de los parámetros operativos (temperaturas, presiones, pH, flujos).
4. **Implementación de Stepwise AIC**:
   - Se utilizó `SequentialFeatureSelector` con scoring de MSE cruzado, en lugar de un algoritmo Stepwise guiado estrictamente por el Criterio de Información de Akaike (AIC/BIC) como exigía el enunciado.
5. **Diagnósticos OLS Incompletos**:
   - No se implementaron pruebas formales de homocedasticidad (Breusch-Pagan), condición de multicolinealidad rigurosa (VIF) ni análisis estructurado de significancia estadística.

---

> **Nota**: Todo el código ha sido completamente rediseñado, refactorizado y profesionalizado en el paquete principal `/src` con patrones de diseño, pipeline reproducible y ejecución modular.
