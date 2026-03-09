"""
Test rápido del modelo circular.

Verifica que el modelo funciona correctamente con:
- Forward pass
- Loss computation
- Angle to phase conversion
- Confidence computation

Uso:
    python scripts/test_circular_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from models.circular_phase_model import (
    build_circular_model,
    angular_loss,
    phase_to_unit_vector,
    angle_to_phase,
    compute_circular_accuracy,
    PHASE_TO_ANGLE,
    PHASE_NAMES,
)


def test_basic_forward():
    """Test forward pass del modelo."""
    print("\n" + "="*60)
    print("TEST 1: Forward pass")
    print("="*60)
    
    # Crear modelo
    model = build_circular_model(architecture="cnn", device="cpu")
    model.eval()
    
    # Input de prueba
    x = torch.randn(4, 15, 128, 128)
    
    # Forward
    with torch.no_grad():
        xy, radius, angle = model(x)
    
    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output xy shape: {xy.shape}")
    print(f"✅ Radius shape: {radius.shape}")
    print(f"✅ Angle shape: {angle.shape}")
    
    assert xy.shape == (4, 2), "XY shape incorrecto"
    assert radius.shape == (4,), "Radius shape incorrecto"
    assert angle.shape == (4,), "Angle shape incorrecto"
    
    print("\n✅ Forward pass OK!")


def test_angle_mapping():
    """Test conversión de fases a ángulos."""
    print("\n" + "="*60)
    print("TEST 2: Phase → Angle → Phase")
    print("="*60)
    
    # Ground truth phases
    true_phases = torch.tensor([0, 1, 2, 0, 1, 2])
    
    # Convertir a unit vectors
    unit_vectors = phase_to_unit_vector(true_phases)
    
    print(f"\nTrue phases: {true_phases.numpy()}")
    print(f"Unit vectors shape: {unit_vectors.shape}")
    
    # Calcular ángulos desde unit vectors
    angles = torch.atan2(unit_vectors[:, 1], unit_vectors[:, 0])
    
    print(f"Angles (rad): {angles.numpy()}")
    print(f"Angles (deg): {np.degrees(angles.numpy())}")
    
    # Convertir de vuelta a fases
    recovered_phases = angle_to_phase(angles)
    
    print(f"Recovered phases: {recovered_phases.numpy()}")
    
    # Verificar
    assert torch.all(true_phases == recovered_phases), "Conversión fase→ángulo→fase falló"
    
    print("\n✅ Angle mapping OK!")


def test_angular_loss():
    """Test función de pérdida angular."""
    print("\n" + "="*60)
    print("TEST 3: Angular Loss")
    print("="*60)
    
    # Caso 1: Predicción perfecta
    true_phases = torch.tensor([0, 1, 2])
    target_xy = phase_to_unit_vector(true_phases)
    
    loss_perfect = angular_loss(target_xy, target_xy)
    print(f"Loss (perfect prediction): {loss_perfect.item():.6f}")
    assert loss_perfect.item() < 1e-6, "Loss perfecta debería ser ~0"
    
    # Caso 2: Predicción totalmente opuesta (180°)
    opposite_xy = -target_xy
    loss_opposite = angular_loss(opposite_xy, target_xy)
    print(f"Loss (opposite prediction): {loss_opposite.item():.6f}")
    assert loss_opposite.item() > 1.9, "Loss opuesta debería ser ~2"
    
    # Caso 3: Predicción a 90°
    perpendicular_xy = torch.stack([target_xy[:, 1], -target_xy[:, 0]], dim=1)
    loss_90 = angular_loss(perpendicular_xy, target_xy)
    print(f"Loss (90° prediction): {loss_90.item():.6f}")
    assert 0.9 < loss_90.item() < 1.1, "Loss a 90° debería ser ~1"
    
    print("\n✅ Angular loss OK!")


def test_confidence():
    """Test cálculo de confianza (radio)."""
    print("\n" + "="*60)
    print("TEST 4: Confidence (Radius)")
    print("="*60)
    
    # Crear modelo
    model = build_circular_model(architecture="cnn", device="cpu")
    model.eval()
    
    # Input de prueba
    x = torch.randn(10, 15, 128, 128)
    
    with torch.no_grad():
        xy, radius, angle = model(x)
    
    print(f"Radius stats:")
    print(f"  Mean: {radius.mean().item():.3f}")
    print(f"  Std: {radius.std().item():.3f}")
    print(f"  Min: {radius.min().item():.3f}")
    print(f"  Max: {radius.max().item():.3f}")
    
    # Verificar que todos los radios son positivos
    assert torch.all(radius >= 0), "Radios deben ser no-negativos"
    
    # Verificar consistencia: radius = sqrt(x² + y²)
    computed_radius = torch.sqrt(xy[:, 0]**2 + xy[:, 1]**2)
    assert torch.allclose(radius, computed_radius, atol=1e-4), "Radio inconsistente"
    
    print("\n✅ Confidence computation OK!")


def test_accuracy_computation():
    """Test cálculo de accuracy."""
    print("\n" + "="*60)
    print("TEST 5: Accuracy Computation")
    print("="*60)
    
    # Predicciones perfectas
    true_phases = torch.tensor([0, 1, 2, 0, 1, 2])
    target_xy = phase_to_unit_vector(true_phases)
    pred_angles = torch.atan2(target_xy[:, 1], target_xy[:, 0])
    
    acc_perfect = compute_circular_accuracy(pred_angles, true_phases)
    print(f"Accuracy (perfect): {acc_perfect:.3f}")
    assert acc_perfect == 1.0, "Accuracy perfecta debería ser 1.0"
    
    # Predicciones malas (todo arterial cuando es venous)
    wrong_phases = torch.zeros_like(true_phases)  # todo arterial
    wrong_xy = phase_to_unit_vector(wrong_phases)
    wrong_angles = torch.atan2(wrong_xy[:, 1], wrong_xy[:, 0])
    
    acc_wrong = compute_circular_accuracy(wrong_angles, true_phases)
    print(f"Accuracy (all arterial): {acc_wrong:.3f}")
    assert acc_wrong < 0.5, "Accuracy mala debería ser baja"
    
    print("\n✅ Accuracy computation OK!")


def test_training_step():
    """Test un paso de training completo."""
    print("\n" + "="*60)
    print("TEST 6: Training Step")
    print("="*60)
    
    # Crear modelo y optimizer
    model = build_circular_model(architecture="cnn", device="cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Batch de prueba
    images = torch.randn(8, 15, 128, 128)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    # Forward
    pred_xy, radius, angle = model(images)
    target_xy = phase_to_unit_vector(labels)
    loss = angular_loss(pred_xy, target_xy)
    
    # Backward
    loss.backward()
    optimizer.step()
    
    print(f"✅ Loss: {loss.item():.4f}")
    print(f"✅ Mean confidence: {radius.mean().item():.3f}")
    
    # Accuracy
    acc = compute_circular_accuracy(angle, labels)
    print(f"✅ Accuracy: {acc:.3f}")
    
    print("\n✅ Training step OK!")


def test_resnet_model():
    """Test modelo ResNet."""
    print("\n" + "="*60)
    print("TEST 7: ResNet Model")
    print("="*60)
    
    # Crear modelo ResNet
    model = build_circular_model(architecture="resnet", pretrained=False, device="cpu")
    model.eval()
    
    # Input de prueba
    x = torch.randn(2, 15, 128, 128)
    
    # Forward
    with torch.no_grad():
        xy, radius, angle = model(x)
    
    print(f"✅ ResNet forward OK")
    print(f"✅ Output xy: {xy.shape}")
    print(f"✅ Radius: {radius.shape}")
    
    assert xy.shape == (2, 2), "ResNet output shape incorrecto"
    
    print("\n✅ ResNet model OK!")


def run_all_tests():
    """Ejecuta todos los tests."""
    print("\n" + "="*60)
    print("🔵 CIRCULAR PHASE MODEL - QUICK TEST")
    print("="*60)
    
    try:
        test_basic_forward()
        test_angle_mapping()
        test_angular_loss()
        test_confidence()
        test_accuracy_computation()
        test_training_step()
        test_resnet_model()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nEl modelo circular está funcionando correctamente.")
        print("Puedes proceder a entrenar con:")
        print("  python scripts/train_circular_model.py")
        
    except AssertionError as e:
        print("\n" + "="*60)
        print(f"❌ TEST FAILED: {e}")
        print("="*60)
        raise
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ ERROR: {e}")
        print("="*60)
        raise


if __name__ == "__main__":
    run_all_tests()
