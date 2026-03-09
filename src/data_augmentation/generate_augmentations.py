"""
Genera augmentaciones físicas basadas en el CSV de splits

Lee el CSV generado por dataset_split.py y crea las augmentaciones
necesarias solo para los registros de training set.
"""

from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
from tqdm import tqdm

from augmentation import get_augmentation_pipeline, get_intensity_augmentation_pipeline


def load_tensor(tensor_path: Path) -> np.ndarray:
    """Carga un tensor desde disco"""
    return np.load(tensor_path)


def apply_augmentation(
    tensors: Dict[str, np.ndarray],
    transform_pipeline,
) -> Dict[str, np.ndarray]:
    """
    Aplica pipeline de augmentación a diccionario de tensores multimodales 2D (C, H, W)
    
    Los tensores están normalizados a [0, 1] (ver extract_slices.prepro).
    
    Args:
        tensors: {phase_name: array} donde array es (C, H, W) normalizado [0,1]
        transform_pipeline: pipeline de albumentations configurado con additional_targets
    
    Returns:
        {phase_name: array_augmented} normalizado [0,1]
    """
    # Convertir a formato albumentations (H, W, C)
    tensors_hwc = {}
    for phase, arr in tensors.items():
        # arr es (C, H, W), transponer a (H, W, C)
        tensors_hwc[phase] = np.transpose(arr, (1, 2, 0))
    
    # Preparar diccionario para transform
    transform_input = {}
    phase_names = list(tensors_hwc.keys())
    
    # Primera fase siempre va como 'image'
    transform_input['image'] = tensors_hwc[phase_names[0]]
    
    # Otras fases van con keys 'image1', 'image2', etc (según additional_targets)
    for i, phase in enumerate(phase_names[1:], start=1):
        transform_input[f'image{i}'] = tensors_hwc[phase]
    
    # Aplicar augmentación
    transformed = transform_pipeline(**transform_input)
    
    # Convertir de vuelta a (C, H, W)
    augmented_volumes = {}
    augmented_volumes[phase_names[0]] = np.transpose(transformed['image'], (2, 0, 1)).astype(np.float32)
    
    for i, phase in enumerate(phase_names[1:], start=1):
        augmented_volumes[phase] = np.transpose(transformed[f'image{i}'], (2, 0, 1)).astype(np.float32)
    
    return augmented_volumes


def generate_augmentations_from_csv(
    splits_csv: Path,
    tensors_root: Path,
    augmentations_output: Path,
    phases: list = ['arterial', 'venous', 'late'],
    verbose: bool = True,
) -> Dict:
    """
    Genera augmentaciones físicas basadas en el CSV de splits
    
    Lee el CSV y genera solo las augmentaciones necesarias para training set:
    - spatial: transformaciones geométricas
    - intensity: transformaciones de intensidad
    - both: spatial + intensity
    
    Args:
        splits_csv: ruta al CSV de splits
        tensors_root: carpeta con tensores raw (data/03_processed/tensors_15ch)
        augmentations_output: carpeta de salida (data/03_processed/augmentations)
        phases: fases a procesar
        verbose: mostrar progress bar
    
    Returns:
        Dict con resumen
    """
    print("\n" + "="*70)
    print("AUGMENTATION GENERATION FROM CSV REGISTRY")
    print("="*70 + "\n")
    
    # Cargar CSV
    print(f"[1/3] Leyendo CSV de splits: {splits_csv}...")
    df_splits = pd.read_csv(splits_csv)
    print(f"      Total registros: {len(df_splits)}\n")
    
    # Filtrar solo training set con augmentaciones (no raw)
    df_train_aug = df_splits[
        (df_splits['split_type'] == 'train') & 
        (df_splits['augmentation_type'] != 'raw')
    ].copy()
    
    print(f"[2/3] Filtrando augmentaciones a generar...")
    print(f"      Total augmentaciones: {len(df_train_aug)}")
    
    # Agrupar por case_id para procesar casos completos
    case_ids_to_process = df_train_aug['case_id'].unique()
    print(f"      Casos únicos: {len(case_ids_to_process)}\n")
    
    # Cargar pipelines de augmentación
    print(f"[3/3] Generando augmentaciones físicas...")
    spatial_pipeline = get_augmentation_pipeline()
    intensity_pipeline = get_intensity_augmentation_pipeline()
    
    augmentations_output = Path(augmentations_output)
    augmentations_output.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'total_cases_processed': 0,
        'total_augmentations_generated': 0,
        'augmentations_by_type': {'spatial': 0, 'intensity': 0, 'both': 0},
    }
    
    # Procesar caso por caso
    progress = tqdm(case_ids_to_process, desc="Casos") if verbose else case_ids_to_process
    
    for case_id in progress:
        # Cargar tensores raw de las 3 fases
        tensors_raw = {}
        case_folder = tensors_root / str(case_id)
        
        all_phases_exist = True
        for phase in phases:
            tensor_path = case_folder / f"{case_id}_{phase}.npy"
            if not tensor_path.exists():
                print(f"\n  [WARN] Falta tensor: {tensor_path}")
                all_phases_exist = False
                break
            tensors_raw[phase] = load_tensor(tensor_path)
        
        if not all_phases_exist:
            continue
        
        # Crear carpeta de salida para este caso
        case_output_folder = augmentations_output / str(case_id)
        case_output_folder.mkdir(parents=True, exist_ok=True)
        
        # Generar las 3 versiones augmentadas
        # 1. Spatial (solo transformaciones geométricas)
        tensors_spatial = apply_augmentation(tensors_raw, spatial_pipeline)
        for phase in phases:
            output_path = case_output_folder / f"{case_id}_{phase}_spatial.npy"
            np.save(output_path, tensors_spatial[phase].astype(np.float32))
            summary['augmentations_by_type']['spatial'] += 1
        
        # 2. Intensity (transformaciones de intensidad + gamma)
        tensors_intensity = apply_augmentation(tensors_raw, intensity_pipeline)
        for phase in phases:
            output_path = case_output_folder / f"{case_id}_{phase}_intensity.npy"
            np.save(output_path, tensors_intensity[phase].astype(np.float32))
            summary['augmentations_by_type']['intensity'] += 1
        
        # 3. Both (spatial primero, luego intensity con gamma)
        tensors_both_temp = apply_augmentation(tensors_spatial, intensity_pipeline)
        for phase in phases:
            output_path = case_output_folder / f"{case_id}_{phase}_both.npy"
            np.save(output_path, tensors_both_temp[phase].astype(np.float32))
            summary['augmentations_by_type']['both'] += 1
        
        summary['total_cases_processed'] += 1
    
    summary['total_augmentations_generated'] = sum(summary['augmentations_by_type'].values())
    
    print(f"\n[OK] Augmentaciones generadas: {summary['total_augmentations_generated']}")
    print(f"    - Spatial: {summary['augmentations_by_type']['spatial']}")
    print(f"    - Intensity: {summary['augmentations_by_type']['intensity']}")
    print(f"    - Both: {summary['augmentations_by_type']['both']}")
    print(f"  Casos procesados: {summary['total_cases_processed']}")
    print(f"  Output: {augmentations_output}\n")
    
    print("="*70 + "\n")
    
    return summary
