"""
Dataset Split para Cross-Validation K-Fold

Divide el dataset en test/train/val y registra en CSV (sin copiar físicamente).

Input: 
- kidney_features.csv en data/02_inter (para obtener case_ids)
- Tensores en data/03_processed/tensors_15ch/

Output:
- CSV con registro de splits (unique_id, tensor_path, phase, case_id, 
  augmentation_type, split_type, fold)
"""

from pathlib import Path
from typing import Dict, List, Tuple
import json
import random
import pandas as pd
from sklearn.model_selection import KFold


def get_unique_case_ids(kidney_features_csv: Path) -> List[str]:
    """
    Extrae lista única de case_ids desde kidney_features.csv
    
    Args:
        kidney_features_csv: ruta al archivo kidney_features.csv
    
    Returns:
        Lista ordenada de case_ids únicos
    """
    df = pd.read_csv(kidney_features_csv)
    case_ids = sorted(df['case_id'].astype(str).unique().tolist())
    return case_ids


def split_dataset(
    case_ids: List[str],
    test_size: float = 0.2,
    n_folds: int = 5,
    random_state: int = 42,
) -> Tuple[List[str], List[List[str]]]:
    """
    Divide case_ids en test set y K folds para cross-validation
    
    Args:
        case_ids: lista de case_ids
        test_size: proporción para test set (0.15-0.20)
        n_folds: número de folds para cross-validation
        random_state: semilla para reproducibilidad
    
    Returns:
        (test_ids, train_folds) donde train_folds es lista de listas
    """
    random.seed(random_state)
    
    # Mezclar case_ids
    case_ids_shuffled = case_ids.copy()
    random.shuffle(case_ids_shuffled)
    
    # Split test set
    n_test = int(len(case_ids_shuffled) * test_size)
    test_ids = case_ids_shuffled[:n_test]
    train_ids = case_ids_shuffled[n_test:]
    
    # K-Fold split del train set
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    train_folds = []
    
    for train_idx, val_idx in kf.split(train_ids):
        fold_train_ids = [train_ids[i] for i in train_idx]
        train_folds.append({
            'train': fold_train_ids,
            'val': [train_ids[i] for i in val_idx]
        })
    
    return test_ids, train_folds


def create_split_csv(
    kidney_features_csv: Path,
    tensors_root: Path,
    output_csv: Path,
    test_size: float = 0.2,
    n_folds: int = 5,
    random_state: int = 42,
    phases: List[str] = ['arterial', 'venous', 'late'],
) -> Dict:
    """
    Crea CSV de splits para cross-validation (sin copiar archivos físicamente)
    
    Cada entrada en el CSV representa un tensor con:
    - unique_id: identificador único (case_id_phase_augtype)
    - tensor_path: ruta al tensor (relativa desde root del proyecto)
    - phase: arterial/venous/late
    - case_id: ID del paciente
    - augmentation_type: raw/spatial/intensity/both
    - split_type: test/train/val
    - fold: número de fold (None para test)
    
    Args:
        kidney_features_csv: ruta a kidney_features.csv
        tensors_root: carpeta con tensores raw (data/03_processed/tensors_15ch)
        output_csv: ruta donde guardar CSV (ej: data/dataset_splits.csv)
        test_size: proporción para test set
        n_folds: número de folds
        random_state: semilla aleatoria
        phases: fases a incluir
    
    Returns:
        Dict con resumen del split
    """
    print("\n" + "="*70)
    print("DATASET SPLIT - CSV REGISTRY (NO PHYSICAL COPIES)")
    print("="*70 + "\n")
    
    # 1. Obtener case_ids
    print("[1/3] Leyendo case_ids desde kidney_features.csv...")
    case_ids = get_unique_case_ids(kidney_features_csv)
    print(f"      Total casos: {len(case_ids)}\n")
    
    # 2. Hacer split
    print(f"[2/3] Dividiendo dataset (test={test_size*100:.0f}%, folds={n_folds})...")
    test_ids, train_folds = split_dataset(
        case_ids, 
        test_size=test_size, 
        n_folds=n_folds, 
        random_state=random_state
    )
    
    print(f"      Test set: {len(test_ids)} casos")
    print(f"      Train set: {len(case_ids) - len(test_ids)} casos")
    print(f"      Folds: {n_folds}")
    
    for fold_idx, fold_info in enumerate(train_folds):
        print(f"        Fold {fold_idx}: train={len(fold_info['train'])}, val={len(fold_info['val'])}")
    print()
    
    # 3. Crear CSV con referencias
    print("[3/3] Generando CSV registry...")
    
    rows = []
    
    # Test set (solo raw, sin augmentaciones)
    for case_id in test_ids:
        for phase in phases:
            tensor_path = f"data/03_processed/tensors_15ch/{case_id}/{case_id}_{phase}.npy"
            rows.append({
                'unique_id': f"{case_id}_{phase}_raw",
                'tensor_path': tensor_path,
                'phase': phase,
                'case_id': case_id,
                'augmentation_type': 'raw',
                'split_type': 'test',
                'fold': None,
            })
    
    # Train/val folds (raw + augmentaciones para train, solo raw para val)
    for fold_idx, fold_info in enumerate(train_folds):
        # Training set: raw + 3 augmentaciones
        for case_id in fold_info['train']:
            for phase in phases:
                # Raw
                tensor_path_raw = f"data/03_processed/tensors_15ch/{case_id}/{case_id}_{phase}.npy"
                rows.append({
                    'unique_id': f"{case_id}_{phase}_raw",
                    'tensor_path': tensor_path_raw,
                    'phase': phase,
                    'case_id': case_id,
                    'augmentation_type': 'raw',
                    'split_type': 'train',
                    'fold': fold_idx,
                })
                
                # Augmentaciones (spatial, intensity, both)
                for aug_type in ['spatial', 'intensity', 'both']:
                    tensor_path_aug = f"data/03_processed/augmentations/{case_id}/{case_id}_{phase}_{aug_type}.npy"
                    rows.append({
                        'unique_id': f"{case_id}_{phase}_{aug_type}",
                        'tensor_path': tensor_path_aug,
                        'phase': phase,
                        'case_id': case_id,
                        'augmentation_type': aug_type,
                        'split_type': 'train',
                        'fold': fold_idx,
                    })
        
        # Validation set: solo raw
        for case_id in fold_info['val']:
            for phase in phases:
                tensor_path = f"data/03_processed/tensors_15ch/{case_id}/{case_id}_{phase}.npy"
                rows.append({
                    'unique_id': f"{case_id}_{phase}_raw",
                    'tensor_path': tensor_path,
                    'phase': phase,
                    'case_id': case_id,
                    'augmentation_type': 'raw',
                    'split_type': 'val',
                    'fold': fold_idx,
                })
    
    # Crear DataFrame y guardar CSV
    df_splits = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_splits.to_csv(output_csv, index=False)
    
    print(f"      [OK] CSV generado: {output_csv}")
    print(f"      Total registros: {len(df_splits)}")
    print(f"        - Test (raw): {len(df_splits[df_splits['split_type']=='test'])}")
    print(f"        - Train (raw+aug): {len(df_splits[(df_splits['split_type']=='train')])}")
    print(f"        - Val (raw): {len(df_splits[df_splits['split_type']=='val'])}")
    print()
    
    # Guardar metadata del split
    split_info = {
        "total_cases": len(case_ids),
        "test_size": test_size,
        "n_folds": n_folds,
        "random_state": random_state,
        "test_cases": test_ids,
        "train_folds": [
            {
                "fold_idx": i,
                "train_cases": fold['train'],
                "val_cases": fold['val'],
                "n_train": len(fold['train']),
                "n_val": len(fold['val']),
            }
            for i, fold in enumerate(train_folds)
        ],
        "csv_output": str(output_csv),
        "total_registry_entries": len(df_splits),
    }
    
    # Guardar JSON junto al CSV
    split_json = output_csv.parent / "split_info.json"
    split_json.write_text(json.dumps(split_info, indent=2))
    print(f"[OK] Metadata guardado en: {split_json}")
    
    print("\n" + "="*70)
    print("RESUMEN")
    print(f"  Total casos: {len(case_ids)}")
    print(f"  Test set: {len(test_ids)} casos ({len(test_ids)/len(case_ids)*100:.1f}%)")
    print(f"  Train set: {len(case_ids) - len(test_ids)} casos")
    print(f"  Folds: {n_folds}")
    print(f"  CSV Output: {output_csv}")
    print(f"  Total registros: {len(df_splits)}")
    print("="*70 + "\n")
    
    return split_info
