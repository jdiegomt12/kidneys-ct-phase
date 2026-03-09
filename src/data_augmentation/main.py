from __future__ import annotations

from pathlib import Path
from typing import Dict

import json

from dataset_split import create_split_csv
from generate_augmentations import generate_augmentations_from_csv


def main(
    *,
    kidney_features_csv: Path = Path("data/02_inter/kidney_features.csv"),
    tensors_source: Path = Path("data/03_processed/tensors_15ch"),
    splits_csv_output: Path = Path("data/dataset_splits.csv"),
    augmentations_output: Path = Path("data/03_processed/augmentations"),
    test_size: float = 0.2,
    n_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Pipeline completo: Dataset Split (CSV) + Generación de Augmentaciones.
    
    PASO 1: Dataset Split por case_id (CSV registry, NO copia física)
    - Divide casos (NO series) en test set (20%) + train folds (5-fold CV)
    - Registra referencias en CSV con columnas:
      * unique_id, tensor_path, phase, case_id, augmentation_type, split_type, fold
    - Input: kidney_features.csv + tensors_15ch/
    - Output: data/dataset_splits.csv
    
    PASO 2: Generación de Augmentaciones SOLO para training sets
    - Lee el CSV y genera físicamente solo las augmentaciones necesarias
    - Genera 3 versiones augmentadas: spatial, intensity, both
    - Raw permanece en data/03_processed/tensors_15ch/ (no se duplica)
    - Output: data/03_processed/augmentations/
    
    Args:
        kidney_features_csv: CSV con case_ids
        tensors_source: Carpeta con tensores raw originales
        splits_csv_output: Ruta donde guardar CSV de splits
        augmentations_output: Carpeta para augmentaciones generadas
        test_size: Proporción para test set
        n_folds: Número de folds para CV
        random_state: Semilla aleatoria
    
    Returns:
        Dict con resumen completo
    """
    
    print("\n" + "="*70)
    print("DATASET PREPARATION PIPELINE - CSV-BASED")
    print("="*70 + "\n")
    
    # ========================================================================
    # PASO 1: DATASET SPLIT POR CASE_ID (CSV REGISTRY)
    # ========================================================================
    print("[1/3] Dataset Split por case_id (CSV registry, no physical copies)...")
    print(f"      Input: {kidney_features_csv}")
    print(f"      Tensors: {tensors_source}")
    print(f"      Output CSV: {splits_csv_output}")
    print(f"      Config: test={test_size*100:.0f}%, folds={n_folds}\n")
    
    split_info = create_split_csv(
        kidney_features_csv=kidney_features_csv,
        tensors_root=tensors_source,
        output_csv=splits_csv_output,
        test_size=test_size,
        n_folds=n_folds,
        random_state=random_state,
    )
    
    print(f"[OK] Split CSV completado: {len(split_info['test_cases'])} casos test, "
          f"{n_folds} folds train\n")
    
    # ========================================================================
    # PASO 2: GENERACIÓN DE AUGMENTACIONES (SOLO TRAINING SETS)
    # ========================================================================
    print("[2/3] Generación de augmentaciones (SOLO training sets)...")
    print(f"      Modo: 3 augmentaciones (spatial, intensity, both)")
    print(f"      Raw: permanece en {tensors_source} (no se duplica)")
    print(f"      Augmentaciones output: {augmentations_output}\n")
    
    aug_summary = generate_augmentations_from_csv(
        splits_csv=splits_csv_output,
        tensors_root=tensors_source,
        augmentations_output=augmentations_output,
        verbose=True,
    )
    
    print(f"[OK] Augmentaciones generadas: {aug_summary['total_augmentations_generated']}\n")
    
    # ========================================================================
    # PASO 3: GUARDAR RESUMEN Y FINALIZAR
    # ========================================================================
    print("[3/3] Guardando resumen...")
    
    summary_data = {
        "split_info": split_info,
        "augmentation": {
            "mode": "csv_based_4versions",
            "versions_generated": ["spatial", "intensity", "both"],
            "raw_location": str(tensors_source),
            "augmentations_location": str(augmentations_output),
            "summary": aug_summary,
        },
        "outputs": {
            "splits_csv": str(splits_csv_output),
            "augmentations_folder": str(augmentations_output),
        }
    }
    
    summary_json = Path(splits_csv_output).parent / "pipeline_summary.json"
    summary_json.write_text(json.dumps(summary_data, indent=2))
    print(f"      [OK] {summary_json}\n")
    
    # Resumen final
    print("="*70)
    print("RESUMEN FINAL")
    print("="*70)
    print(f"\n[SPLIT]")
    print(f"  Total casos: {split_info['total_cases']}")
    print(f"  Test set: {len(split_info['test_cases'])} casos")
    print(f"  Train folds: {n_folds}")
    print(f"  CSV registry: {splits_csv_output}")
    print(f"    Total registros: {split_info['total_registry_entries']}")
    
    print(f"\n[AUGMENTATION]")
    print(f"  Casos procesados: {aug_summary['total_cases_processed']}")
    print(f"  Total augmentaciones: {aug_summary['total_augmentations_generated']}")
    print(f"    - Spatial: {aug_summary['augmentations_by_type']['spatial']}")
    print(f"    - Intensity: {aug_summary['augmentations_by_type']['intensity']}")
    print(f"    - Both: {aug_summary['augmentations_by_type']['both']}")
    
    print(f"\n[OUTPUT]")
    print(f"  Splits CSV: {splits_csv_output}")
    print(f"  Augmentations: {augmentations_output}")
    print(f"  Raw tensors: {tensors_source} (sin duplicar)")
    print("="*70 + "\n")
    
    return summary_data


if __name__ == "__main__":
    summary = main()
