# VANTA — Smart Waste Intelligence Lab

Aplicación de clasificación de residuos con tres checkpoints PyTorch: CNN desde cero, MobileNetV2 y ViT-B/16. La interfaz está en Streamlit y el orden de salida procede de `models/class_names.json`.

## Modelos y resultados

Los checkpoints se buscan en `models/`. Los archivos `models/class_names.json` y `models/comparison_results.csv` contienen los metadatos del proyecto. Los tres archivos `.pth` deben estar en `models/` para que la app pueda hacer inferencia.

El CSV `comparison_results.csv` alimenta Research Lab y Overview. Los resultados mostrados son los valores reportados por ese archivo; la app no los vuelve a calcular.

## Inicio local

1. Clona el repositorio y abre la carpeta en VS Code.
2. Crea el entorno virtual:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Instala las dependencias:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Ejecuta:

   ```powershell
   python -m streamlit run app.py
   ```

5. En Live Vision o Model Arena, selecciona Webcam y permite el acceso a la cámara cuando el navegador lo solicite. También puedes cargar un JPG, PNG o WebP.

## Despliegue en Railway

El proyecto incluye `Procfile`, `runtime.txt` y `.streamlit/config.toml` listos para Railway. Conecta el repositorio de GitHub y Railway detectará la configuración automáticamente.

## Errores frecuentes

- **Checkpoint ausente:** verifica que los tres `.pth` están en `models/`.
- **No se encuentra class_names.json:** restaura el archivo o colócalo en `models/`. Debe listar diez clases en el orden de entrenamiento.
- **Pesos incompatibles:** comprueba que el checkpoint corresponde a la arquitectura declarada y que la versión de torchvision puede construir MobileNetV2/ViT-B/16 con esas capas de salida. El cargador usa `strict=True` para exponer incompatibilidades.
- **torch/torchvision no se importan:** instala versiones compatibles desde el selector de PyTorch dentro del entorno `.venv` activo.
- **Cámara bloqueada:** concede permisos de cámara al navegador/sistema operativo y recarga la pestaña. El cargador de archivos es una alternativa.
- **Primera inferencia lenta o falta de memoria:** la primera predicción carga el checkpoint; ViT-B/16 requiere más memoria. Reiniciar Streamlit libera la memoria de los modelos.
- **No aparecen métricas:** coloca `comparison_results.csv` en `models/`.

La app no entrena modelos automáticamente y no afirma que una predicción sea correcta sin etiqueta de referencia.