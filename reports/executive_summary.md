# 📑 Reporte Ejecutivo: Modelado Predictivo del Consumo de $ClO_2$ en Planta de Celulosa

## 1. Resumen Ejecutivo y Contexto Industrial
En la industria de celulosa en Chile, el blanqueo de pulpa Kraft mediante secuencias multietapa (Pre-Blanqueo $\rightarrow$ D0 $\rightarrow$ EOP $\rightarrow$ D1 $\rightarrow$ D2) representa uno de los mayores costos operativos en reactivos químicos.
- **Límite Operacional de Diseño**: 17.50 kg/ADt.
- **Media Histórica de Consumo Observada**: 19.10 kg/ADt (Máximo: 20.37 kg/ADt).
- **Impacto Económico**: El sobreconsumo genera un costo excesivo de **USD 1,600,000 anuales**. Una optimización del 30% en la eficiencia de reactivos se traduce en un beneficio directo de **USD 500,000 anuales**.

---

## 2. Metodología de Ciencia de Datos y Prevención de Fuga de Datos
1. **Limpieza e Ingestión**: Se procesaron 20,128 registros temporales a intervalos de 2 minutos, eliminando sensores constantes o inhabilitados.
2. **Aislamiento de Data Leakage**: Se excluyeron sumandos directos de la meta para garantizar un modelo fundamentado en variables físicas y termodinámicas del proceso (temperaturas, presiones, pH, brillo y consistencias).
3. **Validación Cruzada y División 80/20**: Semilla fijada estrictamente en `SEED = 2022`, escalando con `StandardScaler` ajustado exclusivamente en el conjunto de entrenamiento.

---

## 3. Tabla Comparativa de Rendimiento de Modelos

| Modelo                        |   Train R2 |   Test R2 |   Test R2 Adj |   Test RMSE (kg/ADt) |   Test MAE (kg/ADt) |      AIC |      BIC |   N Variables | Hiperparametros / Regularizacion                                                                                                                   |
|:------------------------------|-----------:|----------:|--------------:|---------------------:|--------------------:|---------:|---------:|--------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------|
| Regresion Lineal MCO (OLS)    |   0.997357 |  0.997436 |      0.997379 |            0.0551913 |           0.0362802 | -92394.2 | -91717.8 |            87 | N/A                                                                                                                                                |
| Regresion Ridge (L2)          |   0.997357 |  0.997436 |      0.997379 |            0.0551938 |           0.0362767 | -92394.2 | -91717.8 |            87 | regularizacion=L2 (Ridge), lambda_optimo=0.1796, cv_folds=5                                                                                        |
| Regresion Lasso (L1)          |   0.997172 |  0.997248 |      0.997205 |            0.0571784 |           0.0363266 | -91354.4 | -90877.8 |            61 | regularizacion=L1 (Lasso), lambda_optimo=0.0007, num_variables_seleccionadas=61, sparsity_pct=29.8851, cv_folds=5                                  |
| Regresion Elastic Net (L1+L2) |   0.997172 |  0.997248 |      0.997205 |            0.0571784 |           0.0363266 | -91354.4 | -90877.8 |            61 | regularizacion=Elastic Net (L1+L2), lambda_optimo=0.0007, l1_ratio_optimo=1.0000, num_variables_seleccionadas=61, sparsity_pct=29.8851, cv_folds=5 |
| Regresion Stepwise (AIC)      |   0.997023 |  0.997094 |      0.997076 |            0.0587535 |           0.0386478 | -90601.3 | -90401.4 |            25 | criterio=AIC, num_variables_seleccionadas=25, aic_final=-44911.4658                                                                                |

---

## 4. Diagnóstico Econométrico del Modelo Clásico OLS (MCO)
- **Test Durbin-Watson (Autocorrelación de Residuos)**: 1.9683 (Indica presencia de estructura temporal en los residuales de alta frecuencia).
- **Test Jarque-Bera (Normalidad de Residuos)**: Estadístico = 402441.67, p-valor = 0.0000e+00.
- **Test Breusch-Pagan (Homocedasticidad)**: Estadístico = 3007.56, p-valor = 0.0000e+00.
- **Número de Condición**: 307.54 (Evidencia multicolinealidad severa entre sensores acoplados, justificando plenamente el uso de regularización Ridge/Lasso/ElasticNet y selección Stepwise).

---

## 5. Selección del Modelo Óptimo y Conclusiones
- **Modelo Ganador**: **Regresion Stepwise (AIC)**
- **$R^2$ en Test**: 0.9971
- **RMSE en Test**: 0.0588 kg/ADt
- **MAE en Test**: 0.0386 kg/ADt
- **Parsimonia**: 25 variables activas.

Este modelo permite a los operadores de planta anticipar desviaciones respecto al límite de 17.5 kg/ADt con un margen de error inferior a 0.06 kg/ADt, facilitando el control proactivo de las dosificaciones químicas y la captura de ahorros proyectados de hasta USD 500,000 anuales.
