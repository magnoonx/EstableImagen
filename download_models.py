import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {
        "id": "andregn/Realistic_Vision_V6.0-inpainting",
        "folder": "realistic-vision-v6-inpainting",
        "name": "Realistic Vision V6.0 Inpainting",
        "desc": "Fotorrealismo extremo y texturas de piel natural",
        "ignore": ["Realistic_Vision_*.safetensors", "*.pt", "*.fp16.*", "safety_checker/*"]
    },
    {
        "id": "dhanushreddy29/CyberRealistic-v3.3-Inpainting",
        "folder": "cyberrealistic-v3.3-inpainting",
        "name": "CyberRealistic V3.3 Inpainting",
        "desc": "Máxima fidelidad anatómica y arte corporal",
        "ignore": ["safety_checker/*"]
    },
    {
        "id": "Uminosachi/deliberate_v3-inpainting",
        "folder": "deliberate-v3-inpainting",
        "name": "Deliberate V3 Inpainting",
        "desc": "Estilo pictórico de bellas artes, óleo y escultura",
        "ignore": ["safety_checker/*", "*.bin"]
    }
]

def is_model_downloaded(target_dir: Path) -> bool:
    if not target_dir.exists():
        return False
    unet_dir = target_dir / "unet"
    if not unet_dir.exists():
        return False
    return any(unet_dir.glob("diffusion_pytorch_model*"))

def download_all():
    print("=" * 65)
    print("DESCARGA DE MODELOS ALTERNATIVOS LOCALES")
    print("=" * 65)
    
    total = len(MODELS)
    for idx, m in enumerate(MODELS, 1):
        target_dir = MODELS_DIR / m["folder"]
        print(f"\n[{idx}/{total}] Modelo: {m['name']} ({m['desc']})")
        print(f"Destino local: {target_dir}")
        
        if is_model_downloaded(target_dir):
            print(f"-> [OK] Ya se encuentra descargado y listo.")
            continue
            
        print(f"Iniciando descarga desde Hugging Face: {m['id']} ...")
        try:
            snapshot_download(
                repo_id=m["id"],
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                ignore_patterns=m["ignore"]
            )
            print(f"-> [ÉXITO] {m['name']} descargado correctamente.")
        except Exception as e:
            print(f"-> [ERROR] Falló la descarga de {m['name']}: {e}")
            
    print("\n" + "=" * 65)
    print("Todos los modelos solicitados han sido procesados.")
    print("=" * 65)

if __name__ == "__main__":
    download_all()
