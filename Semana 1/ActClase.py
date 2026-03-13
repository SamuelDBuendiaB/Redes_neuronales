import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# 1. Crear el modelo
model = keras.Sequential([
    layers.Dense(1, input_shape=(4,), activation='sigmoid', use_bias=True)
])

# 2. Establecer pesos manualmente
# Pesos: Necesidad=3, Presupuesto=6, Descuento=2, impulsividad=2
# Sesgo: 5
pesos = np.array([[3.0], [2.0], [2.0], [-8.0]])  # shape (4, 1)
sesgo = np.array([1.0])                  # shape (1,)

# Asignar pesos y sesgo a la capa
model.layers[0].set_weights([pesos, sesgo])

# 3. Entrada de prueba (limpieza, menú, sombrero)
entrada = np.array([[0.5, 0.5, 0.3,0.5]])

# 4. Predecir
salida = model.predict(entrada)

# 5. Mostrar resultado
valor = salida[0][0]
print(f"Valor de activación: {valor:.4f}")

# 5. Decisión con 3 opciones
if valor >= 0.7:
    print("Comprar")
elif valor >= 0.4:
    print("Comprar luego")
else:
    print("No comprar")
