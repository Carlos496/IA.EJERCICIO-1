import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

x = np.arange(1, 11, dtype=float)
y = x / 2

model = keras.Sequential([
    layers.Dense(1, input_shape=[1])
])

model.compile(optimizer='sgd', loss='mse')

history = model.fit(x, y, epochs=500, verbose=0)

test_x = np.arange(11, 16, dtype=float)
test_y = test_x / 2
predictions = model.predict(test_x)

print("\nPredicciones vs Valores reales:")
for x_val, pred, real in zip(test_x, predictions, test_y):
    print(f"Entrada: {x_val}, Predicción: {pred[0]:.2f}, Real: {real}")

plt.plot(history.history['loss'])
plt.title('Progreso del entrenamiento')
plt.ylabel('Pérdida')
plt.xlabel('Época')
plt.show()

model.save('mitad_model.h5')
