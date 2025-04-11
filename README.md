# Predicción de Imágenes 

Este proyecto implementa un clasificador de imágenes usando técnicas de Machine Learning con una interfaz web para que los usuarios puedan subir sus propias imágenes y obtener predicciones en tiempo real.

## Tecnologías utilizadas

- **Python**
- **scikit-learn** para los modelos de machine learning
- **Flask** para la aplicación web
- **Pandas y NumPy** para manipulación de datos
- **Matplotlib y Seaborn** para visualizaciones

## Características

- Interfaz web sencilla para subir imágenes
- Clasificación de imágenes en tiempo real
- Visualización de resultados con porcentaje de confianza
- Modelo mejorado con técnicas avanzadas

## Resultados del entrenamiento

Los modelos fueron entrenados y evaluados para asegurar un buen rendimiento en la clasificación de imágenes:

![Comparación de modelos](static/model_comparison.png)

La imagen muestra:
- Una matriz de confusión mostrando el rendimiento por clase
- Comparación entre el modelo base (Random Forest) y el modelo avanzado (Gradient Boosting optimizado)

## Instalación y ejecución
## 1. Clonar el repositorio
git clone https://github.com/TomasMartinez99/prediccion_mnist.git
cd prediccion_mnist

## 2. Crea un entorno virtual
python -m venv venv

## 3. Activa el entorno virtual:
Windows: venv\Scripts\activate
MacOS/Linux: source venv/bin/activate

## 4. Instalar las dependencias:
pip install flask numpy pandas matplotlib seaborn scikit-learn pillow

## 5. Entrenar el modelo:
python train_advanced_model.py

## 6. Ejecutar la aplicación
python app.py