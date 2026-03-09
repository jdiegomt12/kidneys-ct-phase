"""
Script para probar el modelo de clasificación con data minimal.

Genera tensores dummy y entrena/valida el modelo.

Uso:
    python scripts/test_classifier_minimal.py
    python scripts/test_classifier_minimal.py --num-cases 5 --epochs 10
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from supervised import build_model


class DummyDataset(Dataset):
    """Dataset con tensores aleatorios para testing."""
    
    def __init__(self, num_samples: int = 30):
        """
        Crea dataset dummy.
        
        Args:
            num_samples: número de muestras (3 clases balanceadas)
        """
        self.num_samples = num_samples
        
        # Generar tensores aleatorios: (15, 512, 512)
        # 1/3 arterial, 1/3 venous, 1/3 late
        self.data = []
        self.labels = []
        
        for label in range(3):
            for i in range(num_samples // 3):
                # Crear tensor con patrón diferente por fase
                tensor = np.random.randn(15, 512, 512).astype(np.float32)
                
                # Añadir patrón distinto por clase para hacerlo más realista
                tensor += label * 0.1  # Offset diferente por clase
                
                # Normalizar a [0, 1]
                tensor = np.clip(tensor, -1, 1)
                tensor = (tensor + 1) / 2
                
                self.data.append(tensor)
                self.labels.append(label)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'image': torch.from_numpy(self.data[idx]).float(),
            'label': self.labels[idx],
        }


def test_forward_pass(model, device):
    """Test básico de forward pass."""
    print("\n[TEST 1] Forward pass")
    print("="*60)
    
    model.eval()
    
    # Crear batch dummy
    x = torch.randn(4, 15, 512, 512).to(device)
    
    with torch.no_grad():
        y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min().item():.2f}, {y.max().item():.2f}]")
    
    assert y.shape == (4, 3), f"Output shape incorrecto: {y.shape}"
    print("[OK] Forward pass funciona\n")


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entrena 1 epoch."""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        images = batch['image'].to(device)
        labels = torch.tensor(batch['label']).to(device)
        
        # Forward
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * images.size(0)
        _, preds = logits.max(1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Valida el modelo."""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        images = batch['image'].to(device)
        labels = torch.tensor(batch['label']).to(device)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        total_loss += loss.item() * images.size(0)
        _, preds = logits.max(1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(
        description="Test del modelo con data minimal"
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=9,
        help="Numero de muestras de test (default: 9, balanceadas)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Numero de epochs (default: 5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Batch size (default: 3)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda o cpu)"
    )
    
    args = parser.parse_args()
    
    device = args.device
    
    print("\n" + "="*60)
    print("[*] TEST DEL MODELO CON DATA MINIMAL")
    print("="*60)
    print(f"  Device: {device}")
    print(f"  Muestras: {args.num_cases}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print("="*60)
    
    # Crear modelo
    print("\n[*] Creando modelo...")
    model = build_model(num_classes=3, pretrained=False, device=device)
    print(f"  Parametros: {model.get_num_params():,}")
    
    # Test forward pass
    test_forward_pass(model, device)
    
    # Crear dataset dummy
    print("[*] Creando dataset dummy...")
    dataset = DummyDataset(num_samples=args.num_cases)
    print(f"  Muestras: {len(dataset)}")
    
    # Train/val split (70/30)
    split_idx = int(0.7 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(split_idx))
    val_dataset = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Training setup
    print("\n[*] Configurando training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\n[TEST 2] Training loop")
    print("="*60)
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"[Epoch {epoch}/{args.epochs}]")
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.1f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Acc: {val_acc*100:.1f}%")
    
    print("\n" + "="*60)
    print("[OK] TODOS LOS TESTS PASARON")
    print("="*60)
    print("\n[*] El modelo parece funcionar correctamente!")
    print("\nProximos pasos:")
    print("  1. Prueba con tus tensores reales del caso 1")
    print("  2. Aumenta los epochs para mejor convergencia")
    print("  3. Ajusta hiperparametros segun sea necesario")
    print()


if __name__ == "__main__":
    main()
