"""
Script de conveniencia para entrenar el modelo circular de fases.

Uso:
    python scripts/train_circular_model.py
    python scripts/train_circular_model.py --architecture cnn --epochs 50
    python scripts/train_circular_model.py --architecture resnet --batch-size 32
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train_circular_phase import main


if __name__ == "__main__":
    main()
