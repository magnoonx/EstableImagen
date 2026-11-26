# Estudio de Modificación de Imagen con IA (Local y Sin Censura)

Aplicación web 100% privada y local diseñada para modificar y editar imágenes (especialmente fotografía artística, arte corporal y desnudos artísticos) utilizando modelos de difusión sin filtros restrictivos ni dependencias de servicios en la nube.

---

## 🚀 Características Principales

1. **100% Local y Privado**: El modelo (`dreamshaper-8-inpainting`) está alojado en la subcarpeta `models/` dentro del proyecto. Ninguna imagen o texto viaja a servidores externos.
2. **Sin Filtros ni Censura**: La tubería de difusión tiene el `safety_checker` explícitamente desactivado para permitir la creación y edición de desnudos artísticos, anatomía humana y bellas artes sin que las imágenes resulten bloqueadas o en negro.
3. **Triple Modalidad de Trabajo**:
   - **Creación desde Cero (Text-to-Image)**: Deja el lienzo vacío, elige la proporción (cuadrada, vertical u horizontal) y escribe tu descripción (prompt) para generar imágenes artísticas originales desde cero.
   - **Inpainting (Edición de Zonas)**: Pinta con el pincel integrado sobre cualquier parte de la imagen (piel, sombras, vestimenta, fondos, objetos) para modificar o corregir exclusivamente esa zona manteniendo intacto el resto.
   - **Image-to-Image (Transformación General)**: Sube la imagen sin pintar máscara y ajusta el deslizador de *Fuerza de Modificación (Denoising)* para cambiar estilo, iluminación o estética general preservando la pose y anatomía.
4. **Detección Automática de Hardware (CPU y GPU)**:
   - Funciona en CPU con multihilo y algoritmo **DPM++ 2M Karras** (alta calidad en 15 a 20 pasos).
   - Detecta automáticamente tarjetas gráficas NVIDIA (CUDA) para aceleración instantánea. Incluye el script `instalar_soporte_gpu.bat` para activar soporte GPU en 1 clic.

---

## 🛠️ Cómo Iniciar la Aplicación

Puedes iniciar la aplicación de dos formas muy sencillas:

### Opción 1 (Doble Clic):
Haz doble clic sobre el archivo `run.bat` ubicado en la raíz del proyecto.

### Opción 2 (Desde la terminal):
```powershell
.venv\Scripts\python.exe app.py
```

La interfaz web se abrirá automáticamente en tu navegador predeterminado en:
`http://127.0.0.1:7860`

---

## 🎨 Consejos para Edición de Desnudos Artísticos y Bellas Artes

- **Para corregir o retocar piel o detalles:**
  - Pinta una máscara suave sobre la zona deseada con el pincel.
  - En el prompt escribe: `natural skin texture, smooth soft lighting, high fidelity, 8k`.
  - Pasos recomendados: `18`, CFG Scale: `7.0`.

- **Para cambiar iluminación o fondo:**
  - Pinta el fondo alrededor del cuerpo o la silueta.
  - Usa prompts como: `soft dramatic studio lighting, dark minimalist background, volumetric light, fine art portrait`.

- **Para transformar el estilo artístico:**
  - No dibujes ninguna máscara.
  - Ajusta *Fuerza de Modificación* entre `0.30` y `0.45` para no deformar la pose original.
  - Selecciona un preset como *Pintura al Óleo Renacentista* o *Estudio Fotográfico*.

---

## 📁 Estructura del Proyecto

```text
EstableImagen/
│
├── app.py                # Aplicación principal e interfaz web Gradio
├── download_model.py     # Script para descargar y verificar modelos locales
├── run.bat               # Lanzador directo con un clic para Windows
├── requirements.txt      # Dependencias del proyecto
│
├── models/               # Carpeta donde residen los modelos descargados
│   └── dreamshaper-8-inpainting/
│
├── outputs/              # Carpeta donde se guardan automáticamente las imágenes generadas
│
└── .venv/                # Entorno virtual aislado de Python
```

