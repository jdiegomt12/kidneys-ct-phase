# 🔵 Circular Phase Model - Quick Commands

## 📊 Estructura del Tensor

**Input: (B, 15, H, W)**
- 15 canales = 5 slices centrales × 3 neighbors
- Cada slice: [anterior, central, posterior]
- Total: UNA fase (arterial/venous/late) con contexto espacial

---

## 🧪 Testing

### Test completo del modelo
```bash
python scripts/test_circular_model.py
```
✅ Status: PASSED (7/7 tests)

### Test de overfit (sanity check)
```bash
# Replica 1 caso x100 veces - debe alcanzar ~100% accuracy
python scripts/test_overfit_circular.py

# Test solo una fase
python scripts/test_overfit_circular.py --phase arterial --n-copies 200

# Con ResNet
python scripts/test_overfit_circular.py --architecture resnet --epochs 50
```
Si el modelo NO overfittea → hay un bug en el código.

---

## 🎓 Training

### 1. CNN Ligero (rápido)
```bash
python scripts/train_circular_model.py \
    --architecture cnn \
    --epochs 30 \
    --batch-size 16
```
- **Tiempo**: ~5-10 min (GPU)
- **Accuracy esperado**: 80-85%

### 2. ResNet (potente)
```bash
python scripts/train_circular_model.py \
    --architecture resnet \
    --epochs 50 \
    --batch-size 16
```
- **Tiempo**: ~15-20 min (GPU)
- **Accuracy esperado**: 88-92%

### 3. Training sin augmentación
```bash
python scripts/train_circular_model.py \
    --architecture resnet \
    --epochs 30 \
    --no-augment
```

### 4. Training con custom dataset
```bash
python scripts/train_circular_model.py \
    --tensors-root "outputs_full resolution/tensors_15ch" \
    --architecture resnet \
    --epochs 50
```

### 5. Training con CPU (lento)
```bash
python scripts/train_circular_model.py \
    --architecture cnn \
    --epochs 20 \
    --device cpu
```

---

## 📊 Evaluation

### 1. Evaluar modelo entrenado
```bash
python src/evaluate_circular_model.py \
    --model-path models/circular_phase/best_circular_model.pt \
    --architecture resnet
```

### 2. Evaluar y guardar predicciones
```bash
python src/evaluate_circular_model.py \
    --model-path models/circular_phase/best_circular_model.pt \
    --architecture resnet \
    --save-predictions
```

### 3. Evaluar en dataset custom
```bash
python src/evaluate_circular_model.py \
    --model-path models/circular_phase/best_circular_model.pt \
    --tensors-root "outputs_full resolution/tensors_15ch" \
    --architecture resnet
```

---

## 🆚 Comparison

### Comparar Circular vs Softmax
```bash
python scripts/compare_models.py \
    --epochs 30 \
    --batch-size 16
```

Entrena ambos modelos y genera:
- Confusion matrices
- Per-phase accuracy comparison
- Confidence distribution
- JSON con resultados

---

## 🐍 Python API

### Uso básico en notebook/script

```python
from circular_phase_model import (
    build_circular_model,
    angle_to_phase,
    phase_to_unit_vector,
)
import torch

# 1. Cargar modelo
model = build_circular_model(
    architecture="resnet",
    device="cuda"
)

checkpoint = torch.load("models/circular_phase/best_circular_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 2. Inferencia
images = torch.randn(4, 15, 512, 512).cuda()  # Batch de 4 imágenes

with torch.no_grad():
    xy, radius, angle = model(images)
    
    # Clasificación discreta
    phases = angle_to_phase(angle)
    
    print(f"Predicciones: {phases}")
    print(f"Confianza: {radius}")
    print(f"Ángulos: {angle}")

# 3. Filtrar por confianza
confident_mask = radius > 0.7
confident_phases = phases[confident_mask]
print(f"{confident_mask.sum()} predicciones confiables")
```

### Training personalizado

```python
from circular_phase_model import (
    build_circular_model,
    angular_loss,
    phase_to_unit_vector,
    compute_circular_accuracy,
)

# Setup
model = build_circular_model(architecture="resnet", device="cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(epochs):
    for batch in train_loader:
        images = batch['image'].cuda()
        labels = batch['label'].cuda()
        
        # Forward
        pred_xy, radius, angle = model(images)
        target_xy = phase_to_unit_vector(labels)
        
        # Loss
        loss = angular_loss(pred_xy, target_xy)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        acc = compute_circular_accuracy(angle, labels)
        print(f"Loss: {loss.item():.4f}, Acc: {acc:.3f}")
```

---

## 📁 Output Structure

### Training
```
models/circular_phase/
├── best_circular_model.pt       # Mejor modelo
├── last_circular_model.pt        # Último checkpoint
└── training_history.json         # Historial
```

### Evaluation
```
outputs/evaluation_circular/
├── confusion_matrix.png
├── circular_predictions.png
├── confidence_distribution.png
├── evaluation_results.json
└── predictions.npz (opcional)
```

### Comparison
```
outputs/model_comparison/
├── confusion_matrices_comparison.png
├── per_phase_accuracy_comparison.png
├── circular_confidence_distribution.png
└── comparison_results.json
```

---

## 🎯 Recommended Workflow

### 1️⃣ Quick Test (5 min)
```bash
# Verificar que todo funciona
python scripts/test_circular_model.py

# Train rápido con CNN
python scripts/train_circular_model.py --architecture cnn --epochs 10

# Evaluar
python src/evaluate_circular_model.py \
    --model-path models/circular_phase/best_circular_model.pt \
    --architecture cnn
```

### 2️⃣ Full Training (30 min)
```bash
# Train ResNet completo
python scripts/train_circular_model.py \
    --architecture resnet \
    --epochs 50 \
    --batch-size 16

# Evaluar con saving
python src/evaluate_circular_model.py \
    --model-path models/circular_phase/best_circular_model.pt \
    --architecture resnet \
    --save-predictions
```

### 3️⃣ Comparison (60 min)
```bash
# Comparar ambos modelos
python scripts/compare_models.py --epochs 30
```

---

## 🔧 Hyperparameter Tuning

### Learning Rate
```bash
# Bajo (fine-tuning)
python scripts/train_circular_model.py --lr 1e-5 --epochs 50

# Alto (from scratch)
python scripts/train_circular_model.py --lr 1e-3 --epochs 30 --no-pretrained
```

### Batch Size
```bash
# Pequeño (GPU limitada)
python scripts/train_circular_model.py --batch-size 8

# Grande (mejor convergencia)
python scripts/train_circular_model.py --batch-size 32
```

### Weight Decay
```bash
# Más regularización
python scripts/train_circular_model.py --weight-decay 1e-4

# Menos regularización
python scripts/train_circular_model.py --weight-decay 1e-6
```

---

## 📊 Monitoring

### Visualizar training history
```python
import json
import matplotlib.pyplot as plt

with open("models/circular_phase/training_history.json") as f:
    history = json.load(f)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train')
plt.plot(history['val_acc'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png')
```

---

## 🐛 Debug Commands

### Verificar dataset
```python
from phase_dataset import create_dataloaders

train_loader, val_loader = create_dataloaders(
    tensors_root="outputs_x3/tensors_15ch",
    batch_size=2,
)

batch = next(iter(train_loader))
print(f"Image shape: {batch['image'].shape}")
print(f"Label shape: {batch['label'].shape}")
print(f"Labels: {batch['label']}")
```

### Test forward pass
```python
import torch
from circular_phase_model import build_circular_model

model = build_circular_model(architecture="cnn", device="cpu")
x = torch.randn(2, 15, 128, 128)

xy, radius, angle = model(x)
print(f"XY: {xy.shape}")
print(f"Radius: {radius}")
print(f"Angle: {angle}")
```

### Check GPU
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

---

## 📚 Documentation

- **[CIRCULAR_MODEL_GUIDE.md](CIRCULAR_MODEL_GUIDE.md)**: Guía completa
- **[CIRCULAR_MODEL_README.md](CIRCULAR_MODEL_README.md)**: Resumen de implementación
- **[circular_phase_model.py](../src/circular_phase_model.py)**: Código fuente

---

## 🚀 Production Inference

```python
import torch
from circular_phase_model import build_circular_model, angle_to_phase

class PhasePredictor:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model = build_circular_model(architecture="resnet", device=device)
        
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def predict(self, images):
        """
        images: torch.Tensor (B, 15, H, W)
        returns: dict with phases, confidence, angles
        """
        with torch.no_grad():
            images = images.to(self.device)
            xy, radius, angle = self.model(images)
            phases = angle_to_phase(angle)
        
        return {
            'phases': phases.cpu().numpy(),
            'confidence': radius.cpu().numpy(),
            'angles': angle.cpu().numpy(),
        }
    
    def predict_with_filter(self, images, confidence_threshold=0.7):
        """Predice solo casos con alta confianza."""
        results = self.predict(images)
        
        high_conf = results['confidence'] > confidence_threshold
        
        return {
            'phases': results['phases'][high_conf],
            'confidence': results['confidence'][high_conf],
            'n_confident': high_conf.sum(),
            'n_total': len(high_conf),
        }

# Uso
predictor = PhasePredictor("models/circular_phase/best_circular_model.pt")
results = predictor.predict(images)
print(f"Phases: {results['phases']}")
print(f"Confidence: {results['confidence']}")
```

---

**¡Listo para usar!** 🚀

Comienza con:
```bash
python scripts/test_circular_model.py
python scripts/train_circular_model.py --architecture resnet --epochs 50
```
