# train_advanced_model.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml
import pickle
import os

# Crear directorios necesarios
os.makedirs('model', exist_ok=True)
os.makedirs('static', exist_ok=True)

print("Cargando conjunto de datos MNIST...")
# Cargamos MNIST directamente desde OpenML
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
X = X / 255.0  # Normalización
y = y.astype(int)  # Convertir etiquetas a enteros

# Para agilizar el proceso, usaremos una muestra más pequeña
print("Preparando subconjunto de datos para entrenamiento rápido...")
n_samples = 10000  # Usa solo 10,000 muestras para que sea más rápido
indices = np.random.choice(X.shape[0], size=n_samples, replace=False)
X = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
y = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]

# Crear nombres de clase para MNIST (dígitos del 0 al 9)
class_names = [str(i) for i in range(10)]
print(f"Conjunto de datos cargado: {X.shape[0]} imágenes, 10 clases (dígitos 0-9)")

# División en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo base - RandomForest
print("Entrenando modelo base (RandomForest)...")
rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)  # n_jobs=-1 usa todos los núcleos
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Precisión del RandomForest: {rf_accuracy:.4f}")

# Modelo avanzado - Gradient Boosting con búsqueda de hiperparámetros
print("Entrenando modelo avanzado (Gradient Boosting con GridSearch)...")

# Definimos la pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GradientBoostingClassifier())
])

# Parámetros para búsqueda (reducidos para mayor velocidad)
param_grid = {
    'classifier__n_estimators': [50],
    'classifier__learning_rate': [0.1],
    'classifier__max_depth': [3]
}

# GridSearch para encontrar los mejores parámetros
grid_search = GridSearchCV(
    pipeline, param_grid, cv=2, scoring='accuracy', verbose=1
)
grid_search.fit(X_train, y_train)

# Mejor modelo
best_model = grid_search.best_estimator_
print(f"Mejores parámetros: {grid_search.best_params_}")

# Evaluación del modelo avanzado
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo avanzado: {accuracy:.4f}")

# Guardar modelo y nombres de clases
with open('model/advanced_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('model/class_names.pkl', 'wb') as f:
    pickle.dump(class_names, f)

print("Modelo guardado en 'model/advanced_model.pkl'")

# Visualizaciones para el README
plt.figure(figsize=(12, 5))

# Matriz de confusión
plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')

# Comparación de modelos
plt.subplot(1, 2, 2)
models = ['Random Forest', 'Gradient Boosting\n(Optimizado)']
scores = [rf_accuracy, accuracy]
sns.barplot(x=models, y=scores)
plt.title('Comparación de Modelos')
plt.ylim(0, 1.0)
plt.ylabel('Precisión')

# Guardar gráficos
plt.tight_layout()
plt.savefig('static/model_comparison.png')
print("Visualizaciones guardadas en 'static/model_comparison.png'")

# Mostrar algunas imágenes de ejemplo
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test.iloc[i].values.reshape(28, 28), cmap='gray')
    plt.title(f"Real: {y_test.iloc[i]}\nPred: {y_pred[i]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('static/example_predictions.png')
print("Ejemplos guardados en 'static/example_predictions.png'")

# Reporte detallado
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=class_names))