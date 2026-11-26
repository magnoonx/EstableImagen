import os
import sys
import gc
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from diffusers import (
    StableDiffusionInpaintPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler
)
import gradio as gr

# Directorios del proyecto
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Configuración de hilos para CPU
num_cores = os.cpu_count() or 4
torch.set_num_threads(min(num_cores, 8))

# Variables globales para cache de modelos
current_model_name = None
current_pipe = None

def get_available_models():
    """Detecta los modelos descargados en la carpeta models/."""
    available = []
    if MODELS_DIR.exists():
        for item in MODELS_DIR.iterdir():
            if item.is_dir():
                # Verificar si tiene unet o model_index.json
                if (item / "model_index.json").exists() or (item / "unet").exists():
                    available.append(item.name)
            elif item.is_file() and item.suffix in [".safetensors", ".ckpt"]:
                available.append(item.name)
    return available if available else ["dreamshaper-8-inpainting (No descargado aún)"]

def get_device_info():
    """Detecta automáticamente si hay una GPU con soporte CUDA disponible."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return "cuda", torch.float16, f"🚀 GPU: {gpu_name} (CUDA - Aceleración activa)"
    else:
        cores = os.cpu_count() or 4
        return "cpu", torch.float32, f"💻 CPU: {cores} hilos (Modo estándar sin GPU CUDA)"

def load_pipeline(model_name: str):
    """Carga el modelo local seleccionado en memoria con safety checker desactivado."""
    global current_model_name, current_pipe
    
    if current_model_name == model_name and current_pipe is not None:
        return current_pipe
        
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"El modelo '{model_name}' no se encuentra en {MODELS_DIR}. Ejecuta la descarga primero.")
        
    print(f"\n[INFO] Cargando modelo local desde: {model_path}")
    
    # Liberar memoria previa
    if current_pipe is not None:
        del current_pipe
        gc.collect()
        
    device, dtype, device_desc = get_device_info()
    print(f"[INFO] Dispositivo detectado: {device_desc}")
    
    # Cargar pipeline de inpainting sin censura
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        str(model_path),
        safety_checker=None,
        requires_safety_checker=False,
        dtype=dtype,
        local_files_only=True
    )
    
    # Configurar scheduler de alto rendimiento DPM++ 2M Karras (15-20 pasos producen máxima calidad)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="dpmsolver++"
    )
    
    pipe = pipe.to(device)
    
    current_model_name = model_name
    current_pipe = pipe
    
    print(f"[OK] Modelo '{model_name}' cargado con éxito en dispositivo: {device.upper()}")
    return current_pipe

def set_scheduler(pipe, scheduler_type: str):
    """Ajusta el algoritmo de muestreo según la preferencia."""
    if scheduler_type == "DPM++ 2M Karras (Rápido y Nítido, 15-20 pasos)":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++"
        )
    elif scheduler_type == "Euler Ancestral (Creativo y Artístico)":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_type == "DDIM (Determinista)":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

def resize_for_sd(image: Image.Image, max_dim: int = 512):
    """Redimensiona la imagen para que sea múltiplo de 8 y óptima para SD 1.5."""
    w, h = image.size
    scale = min(max_dim / w, max_dim / h)
    new_w = int(round(w * scale / 8)) * 8
    new_h = int(round(h * scale / 8)) * 8
    new_w = max(new_w, 256)
    new_h = max(new_h, 256)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

def extract_mask_from_editor(editor_data):
    """Extrae la imagen base y la máscara trazada en el componente ImageEditor de Gradio."""
    if editor_data is None:
        return None, None
        
    bg = None
    if isinstance(editor_data, dict):
        bg = editor_data.get("background")
        layers = editor_data.get("layers", [])
        composite = editor_data.get("composite")
    else:
        bg = editor_data
        layers = []
        
    if bg is None:
        return None, None
        
    bg = bg.convert("RGB")
    w, h = bg.size
    
    mask = Image.new("L", (w, h), 0)
    has_drawn_strokes = False
    
    if layers:
        for layer in layers:
            if layer is not None:
                layer_rgba = layer.convert("RGBA")
                alpha = layer_rgba.split()[-1]
                alpha_arr = np.array(alpha)
                if np.any(alpha_arr > 10):
                    has_drawn_strokes = True
                    mask_arr = np.maximum(np.array(mask), alpha_arr)
                    mask = Image.fromarray(mask_arr)
                    
    if not has_drawn_strokes:
        return bg, None
        
    # Difuminar ligeramente los bordes de la máscara para una integración suave y natural
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    return bg, mask

def process_image(
    editor_data,
    prompt: str,
    negative_prompt: str,
    model_name: str,
    scheduler_name: str,
    steps: int,
    cfg_scale: float,
    denoising_strength: float,
    seed: int,
    max_resolution: int,
    aspect_ratio: str = "1:1 Cuadrada (512x512)",
    progress=gr.Progress(track_tqdm=True)
):
    """Función principal de generación y edición por IA."""
    if not prompt or not prompt.strip():
        raise gr.Error("Escribe una descripción (prompt) de lo que deseas crear o modificar.")
        
    models = get_available_models()
    if not models or "No descargado" in models[0]:
        raise gr.Error("No se ha encontrado el modelo local descargado. Por favor haz clic en 'Descargar Modelo' primero.")
        
    if model_name not in models:
        model_name = models[0]
        
    pipe = load_pipeline(model_name)
    set_scheduler(pipe, scheduler_name)
    
    # Extraer imagen base y máscara trazada (si se cargó alguna)
    base_image, mask_image = extract_mask_from_editor(editor_data) if editor_data is not None else (None, None)
    
    # Configurar semilla
    if seed is None or seed < 0:
        actual_seed = random.randint(0, 2147483647)
    else:
        actual_seed = int(seed)
        
    device_type = pipe.device.type
    generator = torch.Generator(device=device_type).manual_seed(actual_seed)
    
    # MODO 1: Creación desde cero (Text-to-Image) si no se subió imagen
    if base_image is None:
        mode_used = "Text-to-Image (Creación desde cero)"
        if "Vertical" in aspect_ratio:
            target_size = (512, 768)
        elif "Horizontal" in aspect_ratio:
            target_size = (768, 512)
        else:
            target_size = (max_resolution, max_resolution)
            
        blank_canvas = Image.new("RGB", target_size, (0, 0, 0))
        full_mask = Image.new("L", target_size, 255)
        
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=blank_canvas,
            mask_image=full_mask,
            strength=1.0,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator
        ).images[0]
        orig_w, orig_h = target_size
        
    # MODO 2: Edición de zona marcada (Inpainting)
    elif mask_image is not None:
        mode_used = "Inpainting (Edición de zona marcada)"
        orig_w, orig_h = base_image.size
        resized_base = resize_for_sd(base_image, max_dim=max_resolution)
        resized_mask = mask_image.resize(resized_base.size, Image.Resampling.LANCZOS)
        
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=resized_base,
            mask_image=resized_mask,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator
        ).images[0]
        
    # MODO 3: Transformación general de imagen existente (Img2Img)
    else:
        mode_used = f"Img2Img (Transformación general, Fuerza: {denoising_strength})"
        orig_w, orig_h = base_image.size
        resized_base = resize_for_sd(base_image, max_dim=max_resolution)
        white_mask = Image.new("L", resized_base.size, 255)
        
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=resized_base,
            mask_image=white_mask,
            strength=denoising_strength,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator
        ).images[0]
        
    # Restaurar al tamaño original para preservar la composición original
    final_image = result.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    
    # Guardar copia local en la carpeta outputs
    out_idx = len(list(OUTPUTS_DIR.glob("mod_*.png"))) + 1
    out_file = OUTPUTS_DIR / f"mod_{out_idx:04d}_seed{actual_seed}.png"
    final_image.save(out_file)
    
    info_text = f"**Modo:** {mode_used} | **Semilla:** {actual_seed} | **Pasos:** {steps} | **CFG:** {cfg_scale} | **Guardado:** {out_file.name}"
    return final_image, info_text

def apply_preset(preset_name: str, current_prompt: str):
    """Añade descriptores estéticos artísticos preconfigurados."""
    presets = {
        "Estudio Fotográfico (Iluminación dramática y suave)": "fine art photography, soft studio rim lighting, volumetric shadows, 8k",
        "Piel Realista y Natural": "photorealistic, natural skin texture, warm skin tone, high fidelity, 8k",
        "Pintura al Óleo Renacentista": "classical oil painting, fine art, chiaroscuro, artistic brushstrokes",
        "Mármol Clásico / Escultura": "classical marble sculpture, polished white stone, soft museum gallery light",
        "Iluminación Neón / Cyberpunk": "fine art photography, neon rim lights, cyan magenta glow, dark background"
    }
    addition = presets.get(preset_name, "")
    if not addition:
        return current_prompt
    if not current_prompt.strip():
        return addition
    return f"{current_prompt.strip()}, {addition}"

# Interfaz Web con Gradio con tema adaptativo de alto contraste
custom_theme = gr.themes.Default(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate"
)

custom_css = """
/* Asegurar contraste y legibilidad absoluta en modo claro y modo oscuro */
.main-header {
    text-align: center;
    margin-bottom: 20px;
    padding: 18px 24px;
    border-radius: 12px;
    border: 1px solid var(--border-color-primary);
    background: var(--background-fill-secondary);
}
.main-header h1 {
    font-size: 26px !important;
    margin-bottom: 6px !important;
    font-weight: 700 !important;
}
.main-header p {
    font-size: 14px !important;
}

.panel-box {
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 12px !important;
    padding: 18px !important;
    background: var(--background-fill-secondary) !important;
    margin-bottom: 15px !important;
}

/* ELIMINAR FONDOS PASTEL DE LAS ETIQUETAS Y FORZAR MÁXIMO CONTRASTE */
label > span,
.block-label,
span[data-testid="block-label"],
.label-text {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 3px 0 !important;
    font-size: 14px !important;
    font-weight: 700 !important;
}

/* Colores de texto en Modo Claro */
html:not(.dark) label > span,
html:not(.dark) .block-label,
html:not(.dark) span[data-testid="block-label"],
body:not(.dark) label > span,
body:not(.dark) .block-label {
    color: #0f172a !important;
}
html:not(.dark) .main-header h1,
html:not(.dark) .panel-box h1,
html:not(.dark) .panel-box h2,
html:not(.dark) .panel-box h3,
html:not(.dark) .panel-box strong {
    color: #0f172a !important;
}
html:not(.dark) .main-header p,
html:not(.dark) .panel-box p,
html:not(.dark) .panel-box li,
html:not(.dark) .panel-box span {
    color: #334155 !important;
}
html:not(.dark) .block-info,
html:not(.dark) span[data-testid="block-info"],
html:not(.dark) .info {
    color: #475569 !important;
    font-weight: 500 !important;
}

/* Colores de texto en Modo Oscuro */
html.dark label > span,
html.dark .block-label,
html.dark span[data-testid="block-label"],
body.dark label > span,
body.dark .block-label {
    color: #f8fafc !important;
}
html.dark .main-header h1,
html.dark .panel-box h1,
html.dark .panel-box h2,
html.dark .panel-box h3,
html.dark .panel-box strong {
    color: #ffffff !important;
}
html.dark .main-header p,
html.dark .panel-box p,
html.dark .panel-box li,
html.dark .panel-box span {
    color: #e2e8f0 !important;
}
html.dark .block-info,
html.dark span[data-testid="block-info"],
html.dark .info {
    color: #94a3b8 !important;
    font-weight: 500 !important;
}

/* Botón principal */
.gr-button-primary {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 16px !important;
}
"""

head_script = """
<script>
window.addEventListener("DOMContentLoaded", function() {
    const saved = localStorage.getItem("gradio_theme");
    if (saved === "light") {
        document.documentElement.classList.remove("dark");
        document.body.classList.remove("dark");
    } else if (saved === "dark") {
        document.documentElement.classList.add("dark");
        document.body.classList.add("dark");
    }
});
</script>
"""

with gr.Blocks(title="Estudio de Modificación de Imagen IA Local") as demo:
    _, _, current_device_desc = get_device_info()
    with gr.Column(elem_classes=["main-header"]):
        with gr.Row():
            with gr.Column(scale=9):
                gr.Markdown(
                    f"""
                    # 🎨 Estudio de Modificación de Imagen IA (100% Local & Privado)
                    **Especializado en Arte Corporal, Fotografía Artística e Inpainting sin Censura**  
                    *Todo el procesamiento y los modelos corren localmente en tu equipo. Ninguna imagen o prompt sale a internet.*  
                    **Hardware activo:** `{current_device_desc}`
                    """
                )
            with gr.Column(scale=3, min_width=180):
                theme_btn = gr.Button("🌓 Modo Claro / Oscuro", size="sm")
        
    with gr.Row():
        # Columna Izquierda: Editor de imagen y dibujo de máscara
        with gr.Column(scale=5, elem_classes=["panel-box"]):
            gr.Markdown("### 1. Cargar Imagen y Pintar Máscara (Opcional)")
            gr.Markdown(
                "✨ **Crear desde cero:** Deja este lienzo vacío, escribe tu descripción en el prompt y genera.\n\n"
                "💡 **Modificar una zona específica:** Sube tu imagen y pinta sobre la zona deseada usando el pincel.\n\n"
                "💡 **Transformar foto completa:** Sube la foto sin pintar ninguna máscara y ajusta la *Fuerza de Modificación*."
            )
            
            image_editor = gr.ImageEditor(
                type="pil",
                label="Lienzo de Edición (Opcional para Inpainting / Img2Img)",
                brush=gr.Brush(colors=["#ffffff", "#ff0000", "#00ff00"], default_size=25),
                eraser=gr.Eraser(default_size=25),
                height=480
            )
            
            with gr.Accordion("⚙️ Opciones del Lienzo y Proporción", open=False):
                aspect_ratio = gr.Dropdown(
                    choices=[
                        "1:1 Cuadrada (512x512)",
                        "2:3 Vertical (512x768) - Retratos / Cuerpo Entero",
                        "3:2 Horizontal (768x512) - Paisajes / Estudio"
                    ],
                    value="1:1 Cuadrada (512x512)",
                    label="Proporción (Para Creación desde Cero)"
                )
                max_resolution = gr.Slider(
                    minimum=384,
                    maximum=768,
                    step=64,
                    value=512,
                    label="Resolución de Trabajo IA",
                    info="512px es la resolución nativa de SD 1.5 y la más rápida en CPU. La imagen final se restaura al tamaño original."
                )

        # Columna Derecha: Parámetros e Instrucciones de la IA
        with gr.Column(scale=5, elem_classes=["panel-box"]):
            gr.Markdown("### 2. Instrucciones y Parámetros")
            
            prompt_input = gr.Textbox(
                label="¿Qué deseas crear o modificar? (Prompt en inglés recomendado)",
                placeholder="Ejemplo: smooth skin, soft warm studio rim lighting, delicate lace, 8k photorealistic",
                lines=3
            )
            
            with gr.Row():
                preset_dropdown = gr.Dropdown(
                    choices=[
                        "Seleccionar estilo...",
                        "Estudio Fotográfico (Iluminación dramática y suave)",
                        "Piel Realista y Natural",
                        "Pintura al Óleo Renacentista",
                        "Mármol Clásico / Escultura",
                        "Iluminación Neón / Cyberpunk"
                    ],
                    value="Seleccionar estilo...",
                    label="Añadir Estilo Artístico Rápido"
                )
                apply_preset_btn = gr.Button("Añadir al Prompt", size="sm")
                
            negative_prompt_input = gr.Textbox(
                label="Elementos a Evitar (Negative Prompt)",
                value="ugly, deformed, bad anatomy, bad hands, missing fingers, extra limbs, mutated, disfigured, blurry, low quality, pixelated, artifacts, watermark, signature",
                lines=2
            )
            
            with gr.Row():
                models_list = get_available_models()
                model_dropdown = gr.Dropdown(
                    choices=models_list,
                    value=models_list[0] if models_list else "dreamshaper-8-inpainting",
                    label="Modelo Local"
                )
                scheduler_dropdown = gr.Dropdown(
                    choices=[
                        "DPM++ 2M Karras (Rápido y Nítido, 15-20 pasos)",
                        "Euler Ancestral (Creativo y Artístico)",
                        "DDIM (Determinista)"
                    ],
                    value="DPM++ 2M Karras (Rápido y Nítido, 15-20 pasos)",
                    label="Algoritmo de Muestreo"
                )
                
            with gr.Row():
                steps_slider = gr.Slider(
                    minimum=10,
                    maximum=35,
                    step=1,
                    value=18,
                    label="Pasos de Inferencia (Steps)",
                    info="15-20 pasos ofrecen excelente balance de calidad y velocidad en CPU con DPM++."
                )
                cfg_slider = gr.Slider(
                    minimum=3.0,
                    maximum=15.0,
                    step=0.5,
                    value=7.0,
                    label="Fidelidad al Prompt (CFG Scale)",
                    info="Valores entre 6.0 y 8.0 son ideales."
                )
                
            with gr.Row():
                denoise_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    step=0.05,
                    value=0.45,
                    label="Fuerza de Modificación (Solo para Imagen completa)",
                    info="0.25 = retoques suaves; 0.5 = cambios notables; 0.8 = transformación total."
                )
                seed_input = gr.Number(
                    value=-1,
                    label="Semilla (Seed)",
                    info="-1 para aleatoria en cada ejecución."
                )
                
            generate_btn = gr.Button("🚀 Generar Modificación", variant="primary", size="lg")

    # Fila de Resultados
    with gr.Row(elem_classes=["panel-box"]):
        with gr.Column(scale=6):
            gr.Markdown("### 3. Resultado Modificado")
            output_image = gr.Image(type="pil", label="Imagen Resultante")
            output_info = gr.Markdown("Esperando generación...")
            
        with gr.Column(scale=4):
            gr.Markdown("### 📚 Guía de Uso Rápido")
            gr.Markdown(
                """
                - **Para modificar un detalle (vestimenta, piel, sombras, pose, fondo):**
                  1. Sube tu imagen en el lienzo izquierdo.
                  2. Pinta de blanco o color sobre la parte exacta a cambiar usando el pincel.
                  3. Describe en el prompt lo que debe aparecer en esa zona.
                  4. Haz clic en **Generar Modificación**.
                
                - **Para crear una imagen desde cero (Text-to-Image):**
                  1. Deja el lienzo izquierdo vacío (sin subir imagen).
                  2. En *Opciones del Lienzo*, elige la proporción (cuadrada, vertical u horizontal).
                  3. Describe lo que deseas en el prompt y pulsa **Generar Modificación**.
                
                - **Para alterar la iluminación, estilo o estética general de la foto (Img2Img):**
                  1. Sube la foto y NO dibujes ninguna máscara.
                  2. Ajusta la *Fuerza de Modificación* (0.3 a 0.5 recomendado para preservar la anatomía).
                  3. Escribe el estilo o iluminación deseada.
                
                - **Sin censura:** El modelo no tiene filtros que oculten o arruinen desnudos artísticos.
                """
            )

    # Eventos
    theme_btn.click(
        fn=None,
        js="""
        () => {
            const html = document.documentElement;
            const body = document.body;
            const isDark = html.classList.contains('dark') || body.classList.contains('dark');
            if (isDark) {
                html.classList.remove('dark');
                body.classList.remove('dark');
                localStorage.setItem('gradio_theme', 'light');
            } else {
                html.classList.add('dark');
                body.classList.add('dark');
                localStorage.setItem('gradio_theme', 'dark');
            }
        }
        """
    )

    apply_preset_btn.click(
        fn=apply_preset,
        inputs=[preset_dropdown, prompt_input],
        outputs=[prompt_input]
    )
    
    generate_btn.click(
        fn=process_image,
        inputs=[
            image_editor,
            prompt_input,
            negative_prompt_input,
            model_dropdown,
            scheduler_dropdown,
            steps_slider,
            cfg_slider,
            denoise_slider,
            seed_input,
            max_resolution,
            aspect_ratio
        ],
        outputs=[output_image, output_info]
    )

if __name__ == "__main__":
    print("\nIniciando servidor web local...")
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,
        theme=custom_theme,
        css=custom_css,
        head=head_script
    )

