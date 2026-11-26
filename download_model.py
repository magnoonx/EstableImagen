import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DEFAULT_MODEL_ID = "Lykon/dreamshaper-8-inpainting"
DEFAULT_TARGET_DIR = MODELS_DIR / "dreamshaper-8-inpainting"

def download_dreamshaper():
    print("=" * 60)
    print("DESCARGA DE MODELO LOCAL PARA EDICIÓN ARTÍSTICA")
    print(f"Modelo: {DEFAULT_MODEL_ID}")
    print(f"Destino: {DEFAULT_TARGET_DIR}")
    print("=" * 60)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Comprobar si ya está descargado
    unet_path = DEFAULT_TARGET_DIR / "unet" / "diffusion_pytorch_model.safetensors"
    unet_fp16 = DEFAULT_TARGET_DIR / "unet" / "diffusion_pytorch_model.fp16.safetensors"
    unet_bin = DEFAULT_TARGET_DIR / "unet" / "diffusion_pytorch_model.bin"
    
    if DEFAULT_TARGET_DIR.exists() and (unet_path.exists() or unet_fp16.exists() or unet_bin.exists()):
        print(f"\n[OK] El modelo ya se encuentra descargado en: {DEFAULT_TARGET_DIR}")
        return str(DEFAULT_TARGET_DIR)
        
    print("\nIniciando descarga directa a la carpeta del proyecto...")
    print("Se descargarán únicamente los pesos necesarios (sin duplicados ni filtros).")
    
    # Descargar pesos principales evitando duplicados fp16 y safety_checker
    try:
        snapshot_download(
            repo_id=DEFAULT_MODEL_ID,
            local_dir=str(DEFAULT_TARGET_DIR),
            local_dir_use_symlinks=False,
            ignore_patterns=[
                "*.fp16.safetensors",
                "*.msgpack",
                "*.bin",
                "safety_checker/*"
            ]
        )
        print("\n[ÉXITO] Modelo descargado y listo para uso local offline.")
        return str(DEFAULT_TARGET_DIR)
    except Exception as e:
        print(f"\n[ERROR] Falló la descarga: {e}")
        # En caso de error de filtrado, intentar descarga estándar
        print("Intentando descarga completa...")
        snapshot_download(
            repo_id=DEFAULT_MODEL_ID,
            local_dir=str(DEFAULT_TARGET_DIR),
            local_dir_use_symlinks=False,
            ignore_patterns=["safety_checker/*"]
        )
        return str(DEFAULT_TARGET_DIR)

if __name__ == "__main__":
    download_dreamshaper()

