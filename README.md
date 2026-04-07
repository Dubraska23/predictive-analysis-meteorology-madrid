# 📈 Forecasting de Series Temporales: Análisis Meteorológico Madrid
### Pipeline de Datos con API Meteostat & Modelado ARIMA

Este proyecto implementa un flujo completo de Ciencia de Datos para la obtención, tratamiento y predicción de series temporales meteorológicas. Se destaca por la integración directa con fuentes de datos en tiempo real y un modelado estadístico riguroso para la toma de decisiones basada en proyecciones.

---

## 🛡️ Visión de Gobierno de Datos y Automatización
Como especialista en **Data Governance**, este proyecto ha sido diseñado bajo principios de eficiencia y control de activos de información:

1. **Procedencia del Dato (Data Provenance):** Integración mediante la librería `Meteostat` para asegurar una fuente de datos oficial, veraz y trazable en tiempo real, eliminando los riesgos de manipulación manual de archivos.
2. **Automatización del Ciclo de Vida:** El código gestiona desde la selección de la estación meteorológica por coordenadas hasta la limpieza automatizada de valores nulos (`Interpolación Lineal`), garantizando que la calidad del dato sea apta para el modelado predictivo.
3. **Escalabilidad y Reproducibilidad:** El flujo de trabajo está diseñado para ser replicable en cualquier ubicación geográfica, permitiendo que el proceso de "Data Ingestion" sea un estándar corporativo.
4. **Validación de Integridad:** Se implementan chequeos de estacionalidad y tendencia (Análisis de Residuales) para validar que el modelo predictivo sea estadísticamente sólido antes de generar proyecciones de negocio.

---

## 🎯 Capacidades Técnicas
- **Ingesta de Datos:** Conexión vía API a estaciones meteorológicas internacionales.
- **Análisis de Frecuencias:** Uso de periodogramas y transformada rápida de Fourier para identificar ciclos de estacionalidad.
- **Modelado Predictivo:** Implementación de modelos **ARIMA** (AutoRegressive Integrated Moving Average) para proyecciones a corto plazo.
- **Reconstrucción Armónica:** Modelado de estacionalidad mediante funciones seno/coseno para mayor precisión en la tendencia.

---

## 🛠️ Stack Tecnológico
- **Lenguaje:** Python 3.10+
- **Librerías de API:** `Meteostat`.
- **Modelado Estadístico:** `Statsmodels` (ARIMA, ACF, PACF).
- **Procesamiento de Señal:** `Scipy` (Periodogramas).
- **Análisis y Visualización:** `Pandas`, `Numpy`, `Matplotlib`.

---

## 📊 Resultados: Proyección a 3 días
El modelo entrega una proyección final de **Tmax (Temperatura Máxima)** combinando:
1. Tendencia lineal de largo plazo.
2. Estacionalidad armónica calculada.
3. Residuales proyectados mediante el modelo ARIMA.

---
**Sobre la autora:** Experta en **Data Governance** con formación avanzada en **Business Analytics (UAM)**. Mi enfoque se centra en construir sistemas de datos que no solo sean predictivos, sino también éticos, gobernados y de alta calidad.
