import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np

# Leer el archivo Excel, saltando la primera fila
df = pd.read_csv("data/ganancias.csv", skiprows=1)

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

