# -*- coding: utf-8 -*-
"""
Automatización de augmentación de tensores multifase.
Carga tensores (arterial, venous, late) y aplica augmentación sincronizada.
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
import numpy as np

from augmentation import get_augmentation_pipeline, augment_tensor_multimodal


PHASES = ['arterial', 'venous', 'late']


def find_tensor_cases(tensors_root: Path) -> Dict[str, Path]:
    """
    Encuentra todos los casos con sus 3 fases en tensors_15ch.
    
    Returns:
        Dict[case_id, case_folder_path]
    """
    cases = {}
    tensors_root = Path(tensors_root)
    
    for case_folder in sorted(tensors_root.iterdir()):
        if not case_folder.is_dir():
            continue
        
        case_id = case_folder.name
        
        # Verificar que existan los 3 archivos de fases
        phase_files = {phase: case_folder / f"{case_id}_{phase}.npy" 
                      for phase in PHASES}
        
        if all(f.exists() for f in phase_files.values()):
            cases[case_id] = case_folder
        else:
            missing = [p for p, f in phase_files.items() if not f.exists()]
            print(f"  [WARN] Caso {case_id}: falta {missing}")
    
    return cases


def load_case_tensors(case_folder: Path, case_id: str) -> Dict[str, np.ndarray]:
    """
    Carga los 3 tensores de un caso.
    
    Returns:
        {'arterial': (15,H,W), 'venous': (15,H,W), 'late': (15,H,W)}
    """
    tensors = {}
    for phase in PHASES:
        tensor_path = case_folder / f"{case_id}_{phase}.npy"
        tensors[phase] = np.load(tensor_path)
    return tensors


def augment_case(
    case_id: str,
    tensors: Dict[str, np.ndarray],
    transform,
    num_augmentations: int = 1,
) -> List[Dict[str, np.ndarray]]:
    """
    Aplica augmentación N veces al caso.
    
    Args:
        case_id: identificador del caso
        tensors: dict con los 3 tensores
        transform: pipeline de albumentations
        num_augmentations: cuántas augmentaciones generar
    
    Returns:
        Lista de dicts con tensores augmentados
    """
    augmented_list = []
    
    for aug_idx in range(num_augmentations):
        aug_tensors = augment_tensor_multimodal(tensors, transform)
        augmented_list.append(aug_tensors)
    
    return augmented_list


def save_augmented_tensors(
    case_id: str,
    augmented_list: List[Dict[str, np.ndarray]],
    output_root: Path,
    case_folder: Optional[Path] = None,
) -> List[Path]:
    """
    Guarda los tensores augmentados.
    
    Returns:
        Lista de rutas guardadas
    """
    output_root = Path(output_root)
    saved_paths = []
    
    for aug_idx, aug_tensors in enumerate(augmented_list):
        # Si hay solo 1 augmentación, no añadir sufijo de índice
        suffix = "" if len(augmented_list) == 1 else f"_aug{aug_idx}"
        
        for phase, tensor in aug_tensors.items():
            case_folder_out = output_root / case_id
            case_folder_out.mkdir(parents=True, exist_ok=True)
            
            tensor_path = case_folder_out / f"{case_id}_{phase}{suffix}.npy"
            np.save(tensor_path, tensor.astype(np.float32))
            saved_paths.append(tensor_path)
    
    return saved_paths


def process_tensor_directory(
    tensors_root: Path,
    output_root: Path,
    num_augmentations: int = 1,
    verbose: bool = True,
) -> Dict:
    """
    Automatización completa de augmentación.
    
    Args:
        tensors_root: ruta a carpeta tensors_15ch
        output_root: dónde guardar los tensores augmentados
        num_augmentations: cuántas versiones augmentadas generar por caso
        verbose: mostrar progreso
    
    Returns:
        Dict con resumen de procesamiento
    """
    tensors_root = Path(tensors_root)
    output_root = Path(output_root)
    
    # Encontrar casos
    if verbose:
        print(f"\n[INFO] Buscando casos en {tensors_root}...")
    
    cases = find_tensor_cases(tensors_root)
    
    if not cases:
        print("  [ERROR] No se encontraron casos")
        return {"error": "No cases found", "processed": 0}
    
    if verbose:
        print(f"  [OK] Encontrados {len(cases)} casos\n")
    
    # Preparar pipeline de augmentación
    transform = get_augmentation_pipeline()
    
    # Procesar cada caso
    summary = {
        "total_cases": len(cases),
        "processed": 0,
        "failed": [],
        "total_tensors_saved": 0,
    }
    
    for case_idx, (case_id, case_folder) in enumerate(cases.items(), 1):
        try:
            if verbose:
                print(f"[{case_idx}/{len(cases)}] Procesando caso {case_id}...", end=" ")
            
            # Cargar tensores
            tensors = load_case_tensors(case_folder, case_id)
            
            # Augmentar
            augmented_list = augment_case(
                case_id, tensors, transform,
                num_augmentations=num_augmentations
            )
            
            # Guardar
            saved = save_augmented_tensors(
                case_id, augmented_list, output_root, case_folder
            )
            
            summary["processed"] += 1
            summary["total_tensors_saved"] += len(saved)
            
            if verbose:
                print(f"[OK] ({len(saved)} tensores guardados)")
        
        except Exception as e:
            summary["failed"].append({"case_id": case_id, "error": str(e)})
            if verbose:
                print(f"[ERROR] {e}")
    
    # Resumen final
    if verbose:
        print(f"\n{'='*60}")
        print(f"[OK] Procesados: {summary['processed']}/{len(cases)}")
        print(f"[INFO] Tensores guardados: {summary['total_tensors_saved']}")
        if summary['failed']:
            print(f"[ERROR] Fallos: {len(summary['failed'])}")
            for fail in summary['failed']:
                print(f"   - {fail['case_id']}: {fail['error']}")
        print(f"[INFO] Output: {output_root}")
        print(f"{'='*60}\n")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Augmentación de tensores multifase"
    )
    parser.add_argument(
        "--tensors-root",
        type=str,
        default="outputs_x3/tensors_15ch",
        help="Carpeta con tensores originales"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs_x3/tensors_15ch_augmented",
        help="Carpeta para guardar tensores augmentados"
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=1,
        help="Número de versiones augmentadas por caso"
    )
    
    args = parser.parse_args()
    
    summary = process_tensor_directory(
        tensors_root=args.tensors_root,
        output_root=args.output_root,
        num_augmentations=args.num_augmentations,
        verbose=True,
    )
