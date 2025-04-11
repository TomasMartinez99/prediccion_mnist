from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import io
import base64
import random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Carga el modelo avanzado
with open('model/advanced_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Carga los nombres de clases
with open('model/class_names.pkl', 'rb') as file:
    class_names = pickle.load(file)

def preprocess_image(image_path):
    """
    Preprocesa la imagen para predecir con el modelo MNIST.
    Implementación mejorada que preserva mejor las características.
    """
    # Abre la imagen
    original_img = Image.open(image_path).convert('L')  # Convierte a escala de grises
    
    # Invertir colores si fondo es blanco (asumiendo dígitos oscuros en fondo claro)
    img_array = np.array(original_img)
    if np.mean(img_array) > 128:  # Fondo claro
        img_array = 255 - img_array
        
    # Encuentra bordes del contenido (elimina espacio en blanco)
    non_empty_columns = np.where(img_array.min(axis=0) < 200)[0]
    non_empty_rows = np.where(img_array.min(axis=1) < 200)[0]
    
    if len(non_empty_rows) > 0 and len(non_empty_columns) > 0:
        cropBox = (min(non_empty_columns), min(non_empty_rows),
                  max(non_empty_columns), max(non_empty_rows))
        img_array = img_array[cropBox[1]:cropBox[3]+1, cropBox[0]:cropBox[2]+1]
    
    # Redimensiona a 20x20 conservando proporción y centrando
    # (MNIST usa 20x20 centrado en un campo de 28x28)
    old_size = img_array.shape[:2]
    ratio = float(20) / max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    
    img = Image.fromarray(img_array)
    resized_img = img.resize(new_size[::-1], Image.LANCZOS)  # PIL usa (ancho, alto)
    
    # Centrar en un campo de 28x28
    mnist_img = Image.new('L', (28, 28), 0)  # Fondo negro
    paste_position = ((28 - new_size[1]) // 2, (28 - new_size[0]) // 2)
    mnist_img.paste(resized_img, paste_position)
    
    # Guardar la imagen procesada para visualización
    processed_path = image_path.replace('.', '_processed.')
    mnist_img.save(processed_path)
    
    # Convertir a array plano normalizado como espera el modelo
    final_img_array = np.array(mnist_img).reshape(1, -1) / 255.0
    
    return final_img_array, processed_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    # Guarda la imagen
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Preprocesa y predice
    img_array, processed_path = preprocess_image(filepath)
    
    # Realiza la predicción
    prediction = model.predict(img_array)[0]
    
    # Para modelos que devuelven probabilidades
    try:
        probabilities = model.predict_proba(img_array)[0]
        confidence = round(np.max(probabilities) * 100, 2)
    except:
        confidence = "N/A"
    
    # Extrae solo el nombre del archivo para la plantilla
    processed_filename = os.path.basename(processed_path)
    
    return render_template('resultado.html',
                          imagen=file.filename,
                          imagen_procesada=processed_filename,
                          clase=class_names[int(prediction)],
                          confianza=confidence)

@app.route('/canvas_predict', methods=['POST'])
def canvas_predict():
    # Recibe la imagen del canvas
    canvas_image = request.form.get('image')
    if not canvas_image:
        return redirect(url_for('index'))
    
    # Decodifica la imagen base64
    canvas_image = canvas_image.split(',')[1]
    canvas_data = base64.b64decode(canvas_image)
    
    # Guarda la imagen
    filename = 'canvas_drawing.png'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(filepath, 'wb') as f:
        f.write(canvas_data)
    
    # Preprocesa y predice
    img_array, processed_path = preprocess_image(filepath)
    
    # Realiza la predicción
    prediction = model.predict(img_array)[0]
    
    # Para modelos que devuelven probabilidades
    try:
        probabilities = model.predict_proba(img_array)[0]
        confidence = round(np.max(probabilities) * 100, 2)
    except:
        confidence = "N/A"
    
    # Extrae solo el nombre del archivo para la plantilla
    processed_filename = os.path.basename(processed_path)
    
    return render_template('resultado.html',
                          imagen=filename,
                          imagen_procesada=processed_filename,
                          clase=class_names[int(prediction)],
                          confianza=confidence)

@app.route('/ejemplos')
def ejemplos():
    # Carga algunas imágenes de muestra de MNIST
    from sklearn.datasets import fetch_openml
    
    # Carga una pequeña porción de MNIST
    print("Cargando ejemplos de MNIST...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto', as_frame=False)
    
    # Selecciona 20 imágenes aleatorias (2 de cada dígito)
    ejemplos = []
    for digit in range(10):
        # Encuentra índices donde y es igual al dígito actual
        indices = np.where(np.array(y).astype(int) == digit)[0]
        # Selecciona 2 índices aleatorios
        selected_indices = random.sample(list(indices), 2)
        
        for i in selected_indices:
            # Convierte el array a imagen
            img_array = X[i].reshape(28, 28)
            
            # Guarda la imagen
            img_filename = f"ejemplo_{i}_{y[i]}.png"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
            plt.imsave(filepath, img_array, cmap='gray')
            
            ejemplos.append({
                'filename': img_filename,
                'digit': y[i]
            })
    
    return render_template('ejemplos.html', ejemplos=ejemplos)

@app.route('/predict_ejemplo/<filename>')
def predict_ejemplo(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Preparar la imagen para la predicción (no necesita mucho preprocesamiento ya que son imágenes MNIST)
    img = Image.open(filepath).convert('L')
    img_array = np.array(img).reshape(1, -1) / 255.0
    
    # Predecir
    prediction = model.predict(img_array)[0]
    
    try:
        probabilities = model.predict_proba(img_array)[0]
        confidence = round(np.max(probabilities) * 100, 2)
    except:
        confidence = "N/A"
    
    return render_template('resultado.html',
                          imagen=filename,
                          imagen_procesada=filename,  # Misma imagen
                          clase=class_names[int(prediction)],
                          confianza=confidence,
                          real_digit=filename.split('_')[2].split('.')[0])  # Extrae el dígito real del nombre

if __name__ == '__main__':
    app.run(debug=True)