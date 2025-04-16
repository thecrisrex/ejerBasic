import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np

# Leer el archivo Excel, saltando la primera fila
df = pd.read_csv("data/ganancias.csv", skiprows=1)

import os
os.makedirs("output", exist_ok=True)

# Confirmar columnas
print("Columnas detectadas:", df.columns)

# Transformar de formato ancho a largo
df_largo = df.melt(id_vars="mes", var_name="año", value_name="ganancias")

# Asegurar orden correcto de meses
orden_meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio"]
df_largo["mes"] = pd.Categorical(df_largo["mes"], categories=orden_meses, ordered=True)

# Visualización
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_largo, x="mes", y="ganancias", hue="año", marker="o")
plt.title("Ganancias del 1er Semestre (2021–2023)")
plt.xlabel("Mes")
plt.ylabel("Ganancias")
plt.grid(True)
plt.tight_layout()
plt.savefig("output/lineas.png", dpi=300)
plt.show()



# Agrupamos por año y mes para tener solo una fila por combinación
df_cluster = df_largo.groupby(["año", "mes"], as_index=False,observed=True)["ganancias"].sum()
print("Datos agrupados:\n", df_cluster)
# Clustering usando solo la columna de ganancias
X = df_cluster[["ganancias"]]

# Aplicar KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df_cluster["cluster"] = kmeans.fit_predict(X)

# Visualización
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df_cluster, x="mes", y="ganancias", hue="cluster", palette="Set1")
plt.title("Clustering de Ganancias por Mes (KMeans)")
plt.xlabel("Mes")
plt.ylabel("Ganancias")
plt.grid(True)
plt.tight_layout()
plt.savefig("output/kmeans_clustering.png", dpi=300)
plt.show()

# 1. Gráfico de Barras – Ganancias Totales por Año
total_por_año = df_largo.groupby("año")["ganancias"].sum().reset_index()

plt.figure(figsize=(8, 6))
sns.barplot(data=total_por_año, x="año", y="ganancias", palette="pastel")
plt.title("Ganancias Totales por Año")
plt.xlabel("Año")
plt.ylabel("Ganancia Total")
plt.tight_layout()
plt.savefig("output/barras_totales_por_año.png", dpi=300)
plt.show()

# 2. Heatmap – Ganancias por Mes y Año
tabla_heatmap = df_largo.pivot(index="mes", columns="año", values="ganancias")

plt.figure(figsize=(8, 6))
sns.heatmap(tabla_heatmap, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Ganancias por Mes y Año")
plt.xlabel("Año")
plt.ylabel("Mes")
plt.tight_layout()
plt.savefig("output/heatmap_ganancias.png", dpi=300)
plt.show()

# 3. Boxplot – Distribución de Ganancias por Año
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_largo, x="año", y="ganancias", palette="Set2")
plt.title("Distribución de Ganancias por Año")
plt.xlabel("Año")
plt.ylabel("Ganancias")
plt.tight_layout()
plt.savefig("output/boxplot_ganancias.png", dpi=300)
plt.show()

# 4. Regresión Lineal – Predicción de Ganancias por Año
totales = df_largo.groupby("año")["ganancias"].sum().reset_index()
X = totales["año"].astype(int).values.reshape(-1, 1)
y = totales["ganancias"].values

modelo = LinearRegression()
modelo.fit(X, y)
y_pred = modelo.predict(X)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=totales["año"], y=totales["ganancias"], label="Ganancias reales", s=100)
plt.plot(totales["año"], y_pred, color="red", linestyle="--", label="Regresión Lineal")
plt.title("Regresión Lineal de Ganancias Totales por Año")
plt.xlabel("Año")
plt.ylabel("Ganancias")
plt.legend()
plt.tight_layout()
plt.savefig("output/regresion_lineal.png", dpi=300)
plt.show()

