# PhaseClassifier - Guía de Implementación

## Descripción

Modelo ResNet18 adaptado para clasificar fases de CT multifase (arterial, venous, late) usando tensores de 15 canales (5 slices × 3 fases).

**Arquitectura:**
- Input: (B, 15, 512, 512)
- Convolución inicial: 15 → 64 canales
- ResNet18 blocks
- Global Average Pooling
- FC: 512 → 3 (clases)
- Output: logits para [arterial, venous, late]

## Componentes

### 1. `supervised.py`
Define el modelo `PhaseClassifier`:
```python
from supervised import build_model

model = build_model(num_classes=3, pretrained=True, device="cuda")
print(model.get_num_params())  # Total de parámetros
```

**Características:**
- Adaptación automática de pesos pretrained (ImageNet)
- Primeros 3 canales: copian pesos originales
- Canales 3-15: promedio de pesos

### 2. `phase_dataset.py`
DataLoader que carga tensores NP Y:

**Estructura esperada:**
```
tensors_15ch/
  case_1/
    case_1_arterial.npy
    case_1_venous.npy
    case_1_late.npy
  case_2/
    ...
```

**Uso:**
```python
from phase_dataset import create_dataloaders

train_loader, val_loader = create_dataloaders(
    tensors_root="outputs_x3/tensors_15ch",
    batch_size=16,
    train_split=0.8,
    augment_train=True,  # Augmentación con deformaciones elásticas
    return_all_phases=False,  # 3 samples por caso (1 fase cada uno)
)
```

**Estrategia de labeling:**
- `return_all_phases=False`: Cada muestra es una fase individual
  - 3 samples por caso: arterial (label=0), venous (label=1), late (label=2)
  - Total: N_casos × 3 samples
  
- `return_all_phases=True`: Todas las fases concatenadas (45 canales)
  - 1 sample por caso
  - Total: N_casos samples
  - Útil para aprender representación consolidada

### 3. `train_phase_classifier.py`
Script de training con:
- Learning rate scheduling (Cosine Annealing)
- Checkpointing del mejormodel
- Tracking de historical  - Manejo de validación

**Uso directo:**
```bash
python src/train_phase_classifier.py \
  --tensors-root outputs_x3/tensors_15ch \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-4 \
  --device cuda
```

### 4. `scripts/train_phase_model.py`
Wrapper conveniente con valores por defecto:

**Uso:**
```bash
# Con defaults (30 epochs, batch_size=16, lr=1e-4)
python scripts/train_phase_model.py

# Con argumentos personalizados
python scripts/train_phase_model.py \
  --epochs 50 \
  --batch-size 32 \
  --tensors-root outputs_x3/tensors_15ch \
  --no-augment
```

### 5. `evaluate_phase_classifier.py`
Evalúa modelo entrenado:

**Uso:**
```bash
python src/evaluate_phase_classifier.py \
  --model models/phase_classifier/best_model.pt \
  --tensors-root outputs_x3/tensors_15ch \
  --batch-size 16 \
  --output-dir models/phase_classifier
```

**Output:**
- Accuracy general
- Matriz de confusión
- Precision, Recall, F1 por clase
- Resultados guardados en JSON

## Flujo de trabajo completo

### 1. Preparar datos (tensores)
Los tensores ya deben estar en `outputs_*/tensors_15ch` con estructura:
```
outputs_x3/
  tensors_15ch/
    1/
      1_arterial.npy
      1_venous.npy
      1_late.npy
```

### 2. Augmentación (opcional)
Generar versiones augmentadas:
```bash
python scripts/run_augmentation.py \
  --tensors-root outputs_x3/tensors_15ch \
  --num-augmentations 3
```

Esto crea `tensors_15ch_augmented/` con múltiples versiones de cada fase.

### 3. Entrenar
```bash
# Opción 1: Script simple
python scripts/train_phase_model.py --epochs 50

# Opción 2: Control total
python src/train_phase_classifier.py \
  --tensors-root outputs_x3/tensors_15ch \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-4 \
  --output-dir custom_models/phase \
  --device cuda
```

**Output esperado:**
```
============================================================
[*] Iniciando training por 50 epochs
[*] Device: cuda
[*] Output: models/phase_classifier
============================================================

[Epoch 1/50]
  Train Loss: 1.0875 | Train Acc: 38.46%
  Val Loss:   0.9234 | Val Acc:   52.38%
  [*] Mejor modelo guardado (acc: 52.38%)

[Epoch 2/50]
  Train Loss: 0.8234 | Train Acc: 61.54%
  Val Loss:   0.7123 | Val Acc:   65.08%
  [*] Mejor modelo guardado (acc: 65.08%)
```

**Archivos generados:**
- `models/phase_classifier/best_model.pt` - Mejor modelo
- `models/phase_classifier/last_model.pt` - Último modelo
- `models/phase_classifier/training_history.json` - Historia de loss/acc

### 4. Evaluar
```bash
python src/evaluate_phase_classifier.py \
  --model models/phase_classifier/best_model.pt \
  --tensors-root outputs_x3/tensors_15ch
```

**Output esperado:**
```
============================================================
[RESULTADOS]
============================================================

Accuracy: 92.31%
Loss: 0.2345

[Matriz de confusión]
            arterial   venous      late
arterial           24        1         0
venous              0       23         2
late                1        0        24

[Reporte por clase]
arterial:
  Precision: 0.9600
  Recall:    0.9600
  F1-score:  0.9600
venous:
  Precision: 0.9583
  Recall:    0.9200
  F1-score:  0.9388
late:
  Precision: 0.9231
  Recall:    0.9600
  F1-score:  0.9412
```

## Configuración recomendada

### Para empezar rápido
```bash
python scripts/train_phase_model.py
```

### Para máximo rendimiento (@GPU con mucha VRAM)
```bash
python scripts/train_phase_model.py \
  --epochs 100 \
  --batch-size 64 \
  --lr 5e-5
```

### Para máximo rendimiento (@CPU)
```bash
python scripts/train_phase_model.py \
  --epochs 50 \
  --batch-size 8 \
  --no-augment
```

## Hiperparámetros clave

| Parámetro | Default | Rango recomendado | Notas |
|-----------|---------|-------------------|-------|
| `lr` | 1e-4 | 5e-5 to 5e-4 | Mas bajo = mas lento pero mejor |
| `batch_size` | 16 | 8-64 | Limitar por VRAM disponible |
| `epochs` | 30 | 30-100 | Más con datasets pequeños |
| `weight_decay` | 1e-5 | 1e-6 to 1e-4 | Regularización L2 |

## Troubleshooting

### "CUDA out of memory"
- Reducir `--batch-size` (8 o 4)
- Usar `--device cpu` para probar
- Revisar procesos GPU con `nvidia-smi`

### Loss no baja
- Aumentar `--lr` (1e-3)
- Revisar que datos estén normalizados [0, 1]
- Verificar labels correctos: {0, 1, 2}

### Validation acc muy baja en epoch 1
- Normal en primeras épocas
- Esperar 10-20 epochs para convergencia
- Revisar confusion matrix para bias

## Visualización de resultados

Después de entrenar, puedes visualizar con:
```python
import json
import matplotlib.pyplot as plt

with open("models/phase_classifier/training_history.json") as f:
    history = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history['train_loss'], label='Train')
ax1.plot(history['val_loss'], label='Val')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(history['train_acc'], label='Train')
ax2.plot(history['val_acc'], label='Val')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
```

## Próximos pasos

1. **Augmentación elástica**: Implementar deformaciones no-lineales tipo registro
2. **Multi-task learning**: Predecir fase + características renales simultáneamente
3. **Ensemble**: Combinar múltiples modelos para mejor robustez
4. **Explicability**: Grad-CAM para visualizar regiones determinantes
5. **Quantization**: Convertir a PyTorch mobile/ONNX
