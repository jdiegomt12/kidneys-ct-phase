"""
Script de ejemplo para ejecutar la augmentación de tensores.

Uso:
    python run_augmentation.py
    python run_augmentation.py --tensors-root outputs_x3/tensors_15ch --num-augmentations 3
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from augment_tensors import process_tensor_directory


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Augmentación de tensores multifase CT kidneys",
        epilog="Ejemplos:\n"
               "  python run_augmentation.py\n"
               "  python run_augmentation.py --tensors-root outputs_x3/tensors_15ch --num-augmentations 3\n"
               "  python run_augmentation.py --tensors-root 'outputs_full resolution/tensors_15ch' --num-augmentations 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--tensors-root",
        type=str,
        default="outputs_x3/tensors_15ch",
        help="Ruta a carpeta tensors_15ch (default: outputs_x3/tensors_15ch)"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Ruta para guardar augmentados (por defecto: {tensors-root}_augmented)"
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=1,
        help="Número de versiones augmentadas por caso (default: 1)"
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Sin mensajes de progreso"
    )
    
    args = parser.parse_args()
    
    # Si no especifica output, usar {input}_augmented
    output_root = args.output_root or f"{args.tensors_root}_augmented"
    
    print("\n" + "="*60)
    print("[*] AUGMENTACION DE TENSORES MULTIFASE")
    print("="*60)
    print(f"  Input:  {args.tensors_root}")
    print(f"  Output: {output_root}")
    print(f"  Augmentaciones por caso: {args.num_augmentations}")
    print("="*60 + "\n")
    
    summary = process_tensor_directory(
        tensors_root=args.tensors_root,
        output_root=output_root,
        num_augmentations=args.num_augmentations,
        verbose=not args.no_verbose,
    )
    
    # Retornar código de éxito/error
    return 0 if summary.get("processed", 0) > 0 else 1


if __name__ == "__main__":
    exit(main())
