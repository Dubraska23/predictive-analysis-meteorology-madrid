# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 14:56:38 2025

@author: Dubraska Veroes
"""
!pip install meteostat

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.signal import periodogram, find_peaks
from meteostat import Stations, Daily
from datetime import datetime


# ====================================================================
#%% 0. OBTENCIÓN Y PREPARACIÓN INICIAL DE DATOS (Meteostat)
# ====================================================================

print("--- 0. OBTENCIÓN DE DATOS ---")


# 0.1 Obtener Estación
lat, lon = 40.4183, -3.7028
stations = Stations().nearby(lat, lon)
stations = stations.inventory('daily', (datetime(2018, 1, 1), datetime(2025, 11, 30)))
station = stations.fetch(1)
print(f"Estación seleccionada: \n{station}")

# 0.2 Obtener Datos Diarios
start = datetime(2018, 1, 1)
end = datetime(2025, 11, 30)
data = Daily(station, start, end).fetch()

# 0.3 Inspección inicial de nulos
print("\nNulos antes de la Imputación:")
print(data.isnull().sum())

# ====================================================================
## 1. LIMPIEZA DE DATOS (Interpolación e Imputación)
# ====================================================================

# 1.1 Imputar valores perdidos con interpolación 'spline'
print("\n--- 1. LIMPIEZA Y PREPARACIÓN ---")
columnas_imputar = ['tavg', 'tmin', 'tmax', 'prcp', 'pres']
for col in columnas_imputar:
    # Reemplazar posibles vacíos y Nones por NaN estándar
    data[col] = data[col].replace(['', None], np.nan)
    # Interpolación 'spline' de orden 2
    data[col] = data[col].interpolate(method='spline', order=2)

# 1.2 CREAR DATAFRAME LIMPIO FINAL (df_limpio)
# Convertir el índice (tiempo) en columna y asegurar la selección de todas las variables clave
df_limpio = data.reset_index()

# 1.3 Asegurar el nombre de la columna de tiempo y seleccionar variables clave
try:
    df_limpio = df_limpio[['time', 'tavg', 'tmin', 'tmax']].copy()
except KeyError:
    df_limpio = df_limpio.rename(columns={'index': 'time'})
    df_limpio = df_limpio[['time', 'tavg', 'tmin', 'tmax']].copy()

# 1.4 LIMPIEZA FINAL: Eliminar filas donde tavg sea NaN 
df_limpio = df_limpio.dropna(subset=['tavg']).copy()
df_limpio['time'] = pd.to_datetime(df_limpio['time'])

# 1.5 Preparar las series Z y T 
z = df_limpio['tavg'].values.astype(float)
N = len(z)
t = np.arange(1, N + 1)

print(f"DataFrame final 'df_limpio' con tmin y tmax listo. Longitud de la serie (N): {N}")
print("\nColumnas del DataFrame final:")
print(df_limpio.head())

# ====================================================================
## 2. ANÁLISIS DESCRIPTIVO (Suavizado y Gráfico)
# ====================================================================

# 2.1 Calcular promedios móviles (rolling mean)
data_suavizada_7 = df_limpio.set_index('time')['tavg'].rolling(window=7).mean()
data_suavizada_14 = df_limpio.set_index('time')['tavg'].rolling(window=14).mean()

# 2.2 Gráfico de las series
plt.figure(figsize=(14, 10))
variables = ['tavg', 'tmin', 'tmax']

for i, var in enumerate(variables, 1):
    plt.subplot(3, 1, i)
    plt.plot(df_limpio['time'], df_limpio[var], label=f'{var} Original', color='skyblue', alpha=0.6)
    
    if var == 'tavg':
        plt.plot(df_limpio['time'], data_suavizada_7, label=f'{var} 7 days rolling mean', color='navy', alpha=0.8)
        plt.plot(df_limpio['time'], data_suavizada_14, label=f'{var} 14 days rolling mean', color='blue', alpha=0.8)
    
    plt.title(var.upper())
    plt.xlabel('Fecha')
    plt.ylabel('Valor (°C)')
    plt.legend()

plt.tight_layout()
plt.show()

# ====================================================================
## 3. ELIMINACIÓN DE TENDENCIA (OLS Regresión Lineal)
# ====================================================================

print("\n--- 3. ELIMINACIÓN DE TENDENCIA ---")
X_trend = sm.add_constant(t)
model_trend = sm.OLS(z, X_trend).fit()
trend_hat = model_trend.predict(X_trend)
x = z - trend_hat # Serie detrendida

# Visualización de la Tendencia y Serie Detrendida
plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(df_limpio['time'], z, label='Serie Original (z)', alpha=0.7)
plt.plot(df_limpio['time'], trend_hat, label='Tendencia Ajustada', color='red', linestyle='--')
plt.title('Serie Original y Tendencia Lineal')
plt.ylabel('Tavg (°C)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(df_limpio['time'], x, label='Serie Detrendida (x)', color='green')
plt.axhline(0, color='black', linestyle='-', linewidth=0.8)
plt.title('Serie Detrendida')
plt.xlabel('Fecha')
plt.ylabel('Residuales (°C)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ====================================================================
## 4. ANÁLISIS ESPECTRAL Y DETECCIÓN ARMÓNICA
# ====================================================================

print("\n--- 4. ANÁLISIS ESPECTRAL ---")
# 4.1 Cálculo del Periodograma
freqs, Pxx = periodogram(x)
mask = (freqs > 0) & (freqs <= 0.5)
freqs_use = freqs[mask]
Pxx_use = Pxx[mask]

# 4.2 Detección automática de picos
peak_idx, props = find_peaks(Pxx_use, height=np.quantile(Pxx_use, 0.90), distance=2)
if len(peak_idx) == 0:
    peak_idx, props = find_peaks(Pxx_use, distance=2)

peak_heights = props["peak_heights"]
order = np.argsort(peak_heights)[::-1]
K = 3
main_peaks = peak_idx[order[:K]]
dom_freqs = freqs_use[main_peaks]

print("=== FRECUENCIAS DOMINANTES DETECTADAS ===")
for f in dom_freqs:
    print(f"f = {f:.5f}  -> periodo ≈ {1/f:.2f} pasos")

# 4.3 Gráfico del Periodograma (Escala Logarítmica Corregida)
plt.figure(figsize=(10, 6))
plt.semilogy(freqs_use, Pxx_use, label="Periodograma", color='#4FB3C8')
plt.scatter(freqs_use[main_peaks], Pxx_use[main_peaks], marker='o', color='red', s=50, zorder=5, label="Picos dominantes")
for f, y in zip(dom_freqs, Pxx_use[main_peaks]):
    periodo = 1 / f
    plt.text(f, y, f"T≈{periodo:.1f}", fontsize=9, ha='center', va='bottom', rotation=0, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

plt.title("Periodograma de la Serie Detrendida (Temperatura Diaria)")
plt.xlabel("Frecuencia (ciclos por día)")
plt.ylabel("Densidad espectral de potencia (Escala Logarítmica)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.xlim(0, 0.5)
plt.tight_layout()
plt.show()

# ====================================================================
## 5. AJUSTE DEL MODELO ARMÓNICO Y OBTENCIÓN DE RESIDUALES (u)
# ====================================================================

print("\n--- 5. MODELO ARMÓNICO ---")
# 5.1 Construcción de la Matriz Armónica (X_harm)
X_cols = [np.ones(N)]
for f in dom_freqs:
    X_cols.append(np.cos(2 * np.pi * f * t))
    X_cols.append(np.sin(2 * np.pi * f * t))
X_harm = np.column_stack(X_cols)

# 5.2 Ajuste del modelo OLS
model_harm = sm.OLS(x, X_harm).fit()
x_hat_harm = model_harm.predict(X_harm)
u = x - x_hat_harm # Residuales finales

# 5.3 Visualización de Residuales
plt.figure(figsize=(12, 4))
plt.plot(df_limpio['time'], u, label='Residuales Finales (u)', color='green', linewidth=0.8)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.title('Serie de Residuales (u) - Estacionaria')
plt.xlabel('Fecha')
plt.ylabel('Error (°C)')
plt.legend()
plt.grid(True)
plt.show()

# ====================================================================
## 6. MODELADO ARIMA (ACF/PACF y Ajuste)
# ====================================================================

print("\n--- 6. MODELADO ARIMA ---")
# 6.1 Análisis ACF y PACF
lags = 30
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
plot_acf(u, lags=lags, ax=axes[0], title='Autocorrelación (ACF) de Residuales (u)')
plot_pacf(u, lags=lags, ax=axes[1], title='Autocorrelación Parcial (PACF) de Residuales (u)')
plt.show()

# 6.2 Ajuste del Modelo ARIMA (Ajustar p_val y q_val según los gráficos ACF/PACF)
p_val = 1
d_val = 0
q_val = 1
model_arima = ARIMA(u, order=(p_val, d_val, q_val)).fit()

# ====================================================================
## 7. PROYECCIÓN Y RECONSTRUCCIÓN (3 Días)
# ====================================================================

print("\n--- 7. PROYECCIÓN A 3 DÍAS ---")
steps_ahead = 3
start_forecast = N
end_forecast = N + steps_ahead - 1

# 7.1 Proyectar los Residuales (u)
forecast_u = np.array(model_arima.predict(start=start_forecast, end=end_forecast))

# 7.2 Reconstruir la Estacionalidad y Tendencia
t_forecast = np.arange(start_forecast + 1, end_forecast + 2)
X_trend_forecast = sm.add_constant(t_forecast)
trend_forecast = model_trend.predict(X_trend_forecast)

X_harm_forecast = [np.ones(steps_ahead)]
for f in dom_freqs:
    X_harm_forecast.append(np.cos(2 * np.pi * f * t_forecast))
    X_harm_forecast.append(np.sin(2 * np.pi * f * t_forecast))
X_harm_forecast = np.column_stack(X_harm_forecast)
harm_forecast = model_harm.predict(X_harm_forecast)

# 7.3 Proyección Final
y_forecast = np.array(trend_forecast) + np.array(harm_forecast) + np.array(forecast_u)

print("=== PROYECCIÓN A 3 DÍAS ===")
print("Residuales (u) Proyectados:", forecast_u)
print("Proyección Final (Tavg) a 3 días:", y_forecast)

#%% 1. PREPARACIÓN DE LA SERIE TMAX
# ====================================================================

# 1.1 Extraer la serie Tmax limpia
z_max = df_limpio['tmax'].values.astype(float)
# N = len(z_max)
# t = np.arange(1, N + 1)

# ====================================================================
## 2. ELIMINACIÓN DE TENDENCIA (OLS) PARA TMAX
# ====================================================================

# Vector de tiempo t
X_trend_max = sm.add_constant(t)
model_trend_max = sm.OLS(z_max, X_trend_max).fit()
trend_hat_max = model_trend_max.predict(X_trend_max)

# Serie detrendida para Tmax
x_max = z_max - trend_hat_max

# ====================================================================
## 3. ELIMINACIÓN DE ESTACIONALIDAD ARMÓNICA PARA TMAX
# ====================================================================

model_harm_max = sm.OLS(x_max, X_harm).fit()
x_hat_harm_max = model_harm_max.predict(X_harm)

# Residuales finales para Tmax
u_max = x_max - x_hat_harm_max

# ====================================================================
## 4. MODELADO ARIMA Y PROYECCIÓN A 3 DÍAS
# ====================================================================

# Órdenes ARIMA: Usamos (1, 0, 0) 
p_val = 1
d_val = 0
q_val = 0

# 4.1 Ajuste del Modelo ARIMA sobre los residuales u_max
model_arima_max = ARIMA(u_max, order=(p_val, d_val, q_val)).fit()

# 4.2 Proyección de Residuales (3 días)
steps_ahead = 3
start_forecast = N
end_forecast = N + steps_ahead - 1

forecast_u_max = np.array(model_arima_max.predict(start=start_forecast, end=end_forecast))

# ====================================================================
## 5. RECONSTRUCCIÓN DE LA PROYECCIÓN TMAX
# ====================================================================

# 5.1 Generar el tiempo discreto para la proyección
t_forecast = np.arange(start_forecast + 1, end_forecast + 2)

# 5.2 Reconstruir la Tendencia (Trend)
X_trend_forecast_max = sm.add_constant(t_forecast)
trend_forecast_max = model_trend_max.predict(X_trend_forecast_max)

# 5.3 Reconstruir la Estacionalidad Armónica (Harmonic)
X_harm_forecast = [np.ones(steps_ahead)]
for f in dom_freqs:
    X_harm_forecast.append(np.cos(2 * np.pi * f * t_forecast))
    X_harm_forecast.append(np.sin(2 * np.pi * f * t_forecast))
X_harm_forecast = np.column_stack(X_harm_forecast)
harm_forecast_max = model_harm_max.predict(X_harm_forecast)

# 5.4 Proyección Final (Tmax) = Tendencia + Estacionalidad + Residuales
tmax_forecast = np.array(trend_forecast_max) + np.array(harm_forecast_max) + np.array(forecast_u_max)


# ====================================================================
## 6. IMPRESIÓN Y VISUALIZACIÓN
# ====================================================================

print("\n=== PROYECCIÓN TMAX A 3 DÍAS ===")
print("Residuales (u_max) Proyectados:", forecast_u_max)
print("Proyección Final (Tmax) a 3 días:", tmax_forecast)

# --- Visualización ---

# Fechas para la proyección
last_date = df_limpio['time'].iloc[-1]
projection_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps_ahead, freq='D')

plt.figure(figsize=(14, 6))
plt.plot(df_limpio['time'].iloc[-50:], z_max[-50:], label='Tmax Observada', color='red', alpha=0.7)
plt.plot(projection_dates, tmax_forecast, label='Proyección Tmax (3 Días)', color='darkred', linestyle='--')

plt.title('Proyección de Temperatura Máxima (Tmax)')
plt.xlabel('Fecha')
plt.ylabel('Tmax (°C)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()