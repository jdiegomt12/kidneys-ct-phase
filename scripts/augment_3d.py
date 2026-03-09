"""
Script simple para generar augmentaciones 3D.

Genera campos de deformacion para la region del rinon (z_min a z_max).

Uso:
    python scripts/augment_3d.py
    python scripts/augment_3d.py --num-deformations 4 --max-displacement 4.0
    python scripts/augment_3d.py --volume-shape 150 256 256
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
from augmentation import generate_and_save_deformation_fields


def main():
    parser = argparse.ArgumentParser(
        description="Generar campos de deformación 3D para region del rinon",
        epilog="Ejemplos:\n"
               "  python scripts/augment_3d.py\n"
               "  python scripts/augment_3d.py --num-deformations 4\n"
               "  python scripts/augment_3d.py --max-displacement 5.0\n"
               "  python scripts/augment_3d.py --volume-shape 150 256 256",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--num-deformations",
        type=int,
        default=3,
        help="Numero de campos de deformacion (default: 3)"
    )
    parser.add_argument(
        "--max-displacement",
        type=float,
        default=3.0,
        help="Desplazamiento maximo en voxels (default: 3.0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/deform_fields",
        help="Directorio de output (default: outputs/deform_fields)"
    )
    parser.add_argument(    
        "--volume-shape",
        type=int,
        nargs=3,
        default=[150, 256, 256],
        help="Shape del subvolumen del rinon (D H W). Default: 150 256 256 (region tipica del rinon)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("[*] GENERADOR DE CAMPOS DE DEFORMACION 3D")
    print("="*60)
    print(f"  Num campos: {args.num_deformations}")
    print(f"  Max displacement: {args.max_displacement} voxels")
    print(f"  Volume shape: {args.volume_shape}")
    print(f"  Output: {args.output_dir}")
    print("="*60 + "\n")
    
    fields = generate_and_save_deformation_fields(
        volume_shape=tuple(args.volume_shape),
        num_fields=args.num_deformations,
        output_dir=Path(args.output_dir),
        max_displacement=args.max_displacement,
    )
    
    print(f"\n[OK] {len(fields)} campos generados y guardados")
    print(f"    Ubicacion: {Path(args.output_dir).resolve()}\n")


if __name__ == "__main__":
    main()
