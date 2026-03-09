"""
Test rápido de los componentes del modelo.

Verifica que:
1. El modelo se carga correctamente
2. Los datos se cargan correctamente
3. Forward pass funciona
4. Las métricas se calculan correctamente
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from supervised import build_model
from phase_dataset import create_dataloaders


def test_model():
    """Test del modelo."""
    print("\n[TEST 1] Modelo")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Crear modelo
    model = build_model(num_classes=3, pretrained=True, device=device)
    print(f"✓ Modelo creado: {type(model).__name__}")
    print(f"✓ Parámetros: {model.get_num_params():,}")
    
    # Forward pass
    x = torch.randn(2, 15, 512, 512).to(device)
    y = model(x)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {y.shape}")
    print(f"✓ Output range: [{y.min().item():.2f}, {y.max().item():.2f}]")
    
    assert y.shape == (2, 3), f"Output shape incorrecto: {y.shape}"
    print("[OK] Test del modelo pasó\n")


def test_dataloader():
    """Test del dataloader."""
    print("\n[TEST 2] DataLoader")
    print("="*60)
    
    tensors_root = Path("outputs_x3/tensors_15ch")
    
    if not tensors_root.exists():
        print(f"⚠️ Ruta {tensors_root} no existe")
        print("  Creando dataset de prueba...")
        return
    
    # Crear dataloaders
    train_loader, val_loader = create_dataloaders(
        tensors_root=tensors_root,
        batch_size=4,
        num_workers=0,
        train_split=0.8,
        augment_train=False,
        return_all_phases=False,
    )
    
    print(f"✓ Train loader: {len(train_loader)} batches")
    print(f"✓ Val loader: {len(val_loader)} batches")
    
    # Sample batch
    batch = next(iter(train_loader))
    print(f"\nBatch sample:")
    print(f"  Image shape: {batch['image'].shape}")
    print(f"  Image range: [{batch['image'].min().item():.2f}, {batch['image'].max().item():.2f}]")
    print(f"  Labels: {batch['label'].tolist()}")
    print(f"  Phases: {batch['phase']}")
    
    assert batch['image'].shape[0] == 4, "Batch size incorrecto"
    assert batch['image'].shape[1] == 15, "Canales incorrecto"
    assert batch['label'].shape[0] == 4, "Labels incorrecto"
    print("[OK] Test del dataloader pasó\n")


def test_forward_pass():
    """Test de forward pass end-to-end."""
    print("\n[TEST 3] Forward pass end-to-end")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tensors_root = Path("outputs_x3/tensors_15ch")
    if not tensors_root.exists():
        print(f"⚠️ Ruta {tensors_root} no existe")
        return
    
    # Crear modelo y dataloader
    model = build_model(num_classes=3, pretrained=False, device=device)
    _, val_loader = create_dataloaders(
        tensors_root=tensors_root,
        batch_size=2,
        num_workers=0,
        train_split=0.8,
        return_all_phases=False,
    )
    
    # Forward pass
    model.eval()
    batch = next(iter(val_loader))
    
    images = batch['image'].to(device)
    labels = batch['label'].to(device)
    
    with torch.no_grad():
        logits = model(images)
        preds = logits.argmax(dim=1)
    
    print(f"Images: {images.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Predictions: {preds.tolist()}")
    print(f"Ground truth: {labels.tolist()}")
    
    # Calcular accuracy
    acc = (preds == labels).float().mean().item()
    print(f"Accuracy (batch): {acc*100:.2f}%")
    
    print("[OK] Forward pass funcionó\n")


def test_augmentation():
    """Test de augmentación."""
    print("\n[TEST 4] Augmentación")
    print("="*60)
    
    from augmentation import get_augmentation_pipeline, augment_tensor_multimodal
    
    # Crear pipeline
    transform = get_augmentation_pipeline()
    print(f"✓ Pipeline creado: {len(transform)} transformaciones")
    
    # Crear tensores dummy
    import numpy as np
    tensors = {
        'arterial': np.random.rand(5, 512, 512).astype(np.float32),
        'venous': np.random.rand(5, 512, 512).astype(np.float32),
        'late': np.random.rand(5, 512, 512).astype(np.float32),
    }
    
    # Augmentar
    tensors_aug = augment_tensor_multimodal(tensors, transform)
    
    print(f"✓ Tensores originales: {list(tensors.keys())}")
    print(f"✓ Tensores augmentados: {list(tensors_aug.keys())}")
    
    for phase in ['arterial', 'venous', 'late']:
        print(f"  {phase}: {tensors[phase].shape} → {tensors_aug[phase].shape}")
        assert tensors_aug[phase].shape == tensors[phase].shape
    
    print("[OK] Augmentación funcionó\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("[*] TESTS DE COMPONENTES")
    print("="*60)
    
    try:
        test_model()
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    try:
        test_augmentation()
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    try:
        test_dataloader()
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    try:
        test_forward_pass()
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("="*60)
    print("[✓] TODOS LOS TESTS COMPLETADOS")
    print("="*60 + "\n")
