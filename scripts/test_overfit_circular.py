"""
Test de overfit del modelo circular.

Carga UN SOLO CASO, lo replica x100 y entrena.
El modelo DEBE alcanzar ~100% accuracy (overfitting perfecto).

Si no overfittea → hay un bug en el modelo o training loop.

Uso:
    python scripts/test_overfit_circular.py
    python scripts/test_overfit_circular.py --n-copies 200 --epochs 50
    python scripts/test_overfit_circular.py --architecture cnn
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from circular_phase_model import (
    build_circular_model,
    angular_loss,
    phase_to_unit_vector,
    angle_to_phase,
    compute_circular_accuracy,
    PHASE_NAMES,
)


class OverfitDataset(Dataset):
    """Dataset con un solo caso replicado N veces."""
    
    def __init__(self, tensor: np.ndarray, label: int, n_copies: int = 100):
        """
        Args:
            tensor: (15, H, W) numpy array
            label: fase {0, 1, 2}
            n_copies: número de réplicas
        """
        self.tensor = torch.from_numpy(tensor).float()
        self.label = label
        self.n_copies = n_copies
    
    def __len__(self):
        return self.n_copies
    
    def __getitem__(self, idx):
        # Retornar siempre el mismo tensor
        return {
            'image': self.tensor,
            'label': torch.tensor(self.label, dtype=torch.long)
        }


def load_single_case(tensors_root: Path, case_id: str = None):
    """
    Carga un caso aleatorio de la carpeta.
    
    Returns:
        dict con las 3 fases {arterial, venous, late}
    """
    tensors_root = Path(tensors_root)
    
    # Encontrar casos disponibles
    case_folders = [f for f in tensors_root.iterdir() if f.is_dir()]
    
    if not case_folders:
        raise ValueError(f"No se encontraron casos en {tensors_root}")
    
    # Seleccionar caso
    if case_id is None:
        case_folder = case_folders[0]
        case_id = case_folder.name
    else:
        case_folder = tensors_root / case_id
    
    if not case_folder.exists():
        raise ValueError(f"Case {case_id} no existe en {tensors_root}")
    
    print(f"\n📂 Cargando caso: {case_id}")
    
    # Cargar las 3 fases
    phases = {}
    for phase_idx, phase_name in enumerate(PHASE_NAMES):
        path = case_folder / f"{case_id}_{phase_name}.npy"
        
        if not path.exists():
            raise ValueError(f"No se encontró {path}")
        
        tensor = np.load(path)
        phases[phase_name] = {
            'tensor': tensor,
            'label': phase_idx,
            'path': path,
        }
        print(f"  ✅ {phase_name}: {tensor.shape}")
    
    return phases, case_id


def test_overfit_single_phase(
    tensor: np.ndarray,
    label: int,
    phase_name: str,
    n_copies: int,
    epochs: int,
    architecture: str,
    device: str,
    batch_size: int = 16,
):
    """
    Testa overfit de una sola fase.
    """
    print("\n" + "="*60)
    print(f"🔬 OVERFIT TEST: {phase_name.upper()}")
    print("="*60)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Label: {label} ({phase_name})")
    print(f"Copies: {n_copies}")
    print(f"Epochs: {epochs}")
    print("="*60)
    
    # Crear dataset
    dataset = OverfitDataset(tensor, label, n_copies=n_copies)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"\n📊 Dataset: {len(dataset)} samples (all identical)")
    print(f"   Batches: {len(dataloader)}")
    
    # Crear modelo
    print("\n🧠 Creating model...")
    model = build_circular_model(architecture=architecture, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    print(f"\n🎓 Training for {epochs} epochs...")
    
    history = {'loss': [], 'acc': [], 'confidence': []}
    
    for epoch in range(epochs):
        model.train()
        
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        all_radius = []
        
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward
            optimizer.zero_grad()
            pred_xy, radius, angle = model(images)
            target_xy = phase_to_unit_vector(labels)
            loss = angular_loss(pred_xy, target_xy)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Metrics
            epoch_loss += loss.item() * images.size(0)
            pred_phases = angle_to_phase(angle)
            epoch_correct += (pred_phases == labels).sum().item()
            epoch_total += labels.size(0)
            all_radius.extend(radius.detach().cpu().numpy())
        
        avg_loss = epoch_loss / epoch_total
        avg_acc = epoch_correct / epoch_total
        avg_confidence = np.mean(all_radius)
        
        history['loss'].append(avg_loss)
        history['acc'].append(avg_acc)
        history['confidence'].append(avg_confidence)
        
        # Print cada 5 epochs o último
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d}/{epochs}: Loss={avg_loss:.4f}, Acc={avg_acc:.3f} ({avg_acc*100:.1f}%), Conf={avg_confidence:.3f}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        images = torch.stack([dataset[i]['image'] for i in range(min(50, len(dataset)))]).to(device)
        labels = torch.tensor([dataset[i]['label'] for i in range(min(50, len(dataset)))]).to(device)
        
        pred_xy, radius, angle = model(images)
        pred_phases = angle_to_phase(angle)
        
        final_acc = (pred_phases == labels).float().mean().item()
        final_conf = radius.mean().item()
        
        print(f"Final Accuracy: {final_acc:.3f} ({final_acc*100:.1f}%)")
        print(f"Final Confidence: {final_conf:.3f}")
        print(f"Predictions: {pred_phases[:10].cpu().numpy()}")
        print(f"True labels: {labels[:10].cpu().numpy()}")
    
    # Verificar overfit
    print("\n" + "="*60)
    if final_acc >= 0.95:
        print("✅ OVERFIT SUCCESSFUL! El modelo puede aprender.")
        success = True
    elif final_acc >= 0.80:
        print("⚠️  OVERFIT PARCIAL. El modelo aprende pero no perfectamente.")
        print("   → Prueba más epochs o learning rate más alto")
        success = False
    else:
        print("❌ OVERFIT FAILED! Algo está mal.")
        print("   → Revisar modelo, loss, o training loop")
        success = False
    print("="*60)
    
    return {
        'phase': phase_name,
        'final_acc': final_acc,
        'final_conf': final_conf,
        'history': history,
        'success': success,
    }


def test_overfit_all_phases(
    phases: dict,
    n_copies: int,
    epochs: int,
    architecture: str,
    device: str,
):
    """
    Testa overfit de las 3 fases.
    """
    print("\n" + "="*60)
    print("🔬 OVERFIT TEST - ALL PHASES")
    print("="*60)
    
    results = {}
    
    for phase_name in PHASE_NAMES:
        result = test_overfit_single_phase(
            tensor=phases[phase_name]['tensor'],
            label=phases[phase_name]['label'],
            phase_name=phase_name,
            n_copies=n_copies,
            epochs=epochs,
            architecture=architecture,
            device=device,
        )
        results[phase_name] = result
    
    # Summary
    print("\n\n" + "="*60)
    print("📊 SUMMARY - ALL PHASES")
    print("="*60)
    
    all_success = True
    for phase_name in PHASE_NAMES:
        r = results[phase_name]
        status = "✅" if r['success'] else "❌"
        print(f"{status} {phase_name.upper():8s}: Acc={r['final_acc']:.3f} ({r['final_acc']*100:.1f}%), Conf={r['final_conf']:.3f}")
        all_success = all_success and r['success']
    
    print("="*60)
    
    if all_success:
        print("\n🎉 ALL PHASES OVERFITTED SUCCESSFULLY!")
        print("   El modelo circular funciona correctamente.")
    else:
        print("\n⚠️  SOME PHASES FAILED TO OVERFIT")
        print("   Revisar configuración o aumentar epochs.")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test de overfit del modelo circular"
    )
    
    parser.add_argument(
        "--tensors-root",
        type=str,
        default="outputs_x3/tensors_15ch",
        help="Ruta a carpeta tensors_15ch"
    )
    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help="ID del caso a usar (default: primer caso encontrado)"
    )
    parser.add_argument(
        "--n-copies",
        type=int,
        default=100,
        help="Número de réplicas del caso (default: 100)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Número de epochs (default: 30)"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="cnn",
        choices=["cnn", "resnet"],
        help="Arquitectura (default: cnn)"
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["arterial", "venous", "late", "all"],
        help="Fase a testear (default: all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🔬 CIRCULAR MODEL - OVERFIT TEST")
    print("="*60)
    print(f"Tensors root: {args.tensors_root}")
    print(f"Case ID: {args.case_id or 'auto'}")
    print(f"Copies: {args.n_copies}")
    print(f"Epochs: {args.epochs}")
    print(f"Architecture: {args.architecture}")
    print(f"Phase: {args.phase}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # Cargar caso
    phases, case_id = load_single_case(args.tensors_root, args.case_id)
    
    # Test según fase seleccionada
    if args.phase == "all":
        results = test_overfit_all_phases(
            phases=phases,
            n_copies=args.n_copies,
            epochs=args.epochs,
            architecture=args.architecture,
            device=args.device,
        )
    else:
        result = test_overfit_single_phase(
            tensor=phases[args.phase]['tensor'],
            label=phases[args.phase]['label'],
            phase_name=args.phase,
            n_copies=args.n_copies,
            epochs=args.epochs,
            architecture=args.architecture,
            device=args.device,
        )
        results = {args.phase: result}
    
    print("\n✅ Overfit test completed!")


if __name__ == "__main__":
    main()
