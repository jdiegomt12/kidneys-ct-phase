"""
Script de conveniencia para entrenar el PhaseClassifier.

Uso:
    python scripts/train_phase_model.py
    python scripts/train_phase_model.py --epochs 100 --batch-size 32
    python scripts/train_phase_model.py --tensors-root outputs_full\ resolution/tensors_15ch
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
from train_phase_classifier import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="[*] Entrenar PhaseClassifier",
        epilog="Ejemplos:\n"
               "  python scripts/train_phase_model.py\n"
               "  python scripts/train_phase_model.py --epochs 50 --batch-size 32\n"
               "  python scripts/train_phase_model.py --tensors-root 'outputs_full resolution/tensors_15ch'",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--tensors-root",
        type=str,
        default="outputs_x3/tensors_15ch",
        help="Ruta a carpeta tensors_15ch (default: outputs_x3/tensors_15ch)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Numero de epochs (default: 30)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/phase_classifier",
        help="Directorio de output (default: models/phase_classifier)"
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Sin augmentacion en training"
    )
    
    args = parser.parse_args()
    
    # Transferir argumentos al script principal
    sys.argv = [
        sys.argv[0],
        "--tensors-root", args.tensors_root,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--output-dir", args.output_dir,
    ]
    
    if args.no_augment:
        sys.argv.append("--no-augment")
    
    # Ejecutar training
    main()
