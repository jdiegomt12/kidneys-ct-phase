# 🔵 Circular Phase Classification Model - Guía de Uso

## 📋 Índice

1. [Introducción](#introducción)
2. [Instalación](#instalación)
3. [Quick Start](#quick-start)
4. [Arquitectura del Modelo](#arquitectura-del-modelo)
5. [Entrenamiento](#entrenamiento)
6. [Evaluación](#evaluación)
7. [Ventajas vs Softmax](#ventajas-vs-softmax)
8. [Casos de Uso Avanzados](#casos-de-uso-avanzados)

---

## 🎯 Introducción

El **Circular Phase Classification Model** trata la clasificación de fases de contraste CT como un problema de **regresión circular** en lugar de clasificación discreta tradicional.

### Estructura del Input

El tensor de entrada tiene **15 canales** que representan:
- **5 slices centrales** del riñón
- Cada slice con **2 vecinos** (anterior y posterior)
- Total: **5 × 3 = 15 canales** para **UNA fase** específica

Ejemplo:
```
canal 0-2:   slice 1 + vecinos (anterior, central, posterior)
canal 3-5:   slice 2 + vecinos  
canal 6-8:   slice 3 + vecinos
canal 9-11:  slice 4 + vecinos
canal 12-14: slice 5 + vecinos
```

**El modelo predice qué fase es ese tensor:** arterial, venous o late.

### Dataset
Cada caso tiene 3 archivos `.npy`:
- `case_1_arterial.npy` → (15, H, W)
- `case_1_venous.npy` → (15, H, W)
- `case_1_late.npy` → (15, H, W)

Cada archivo es un sample independiente con su label (0, 1, 2).

### ¿Por qué circular?

Las fases CT son **cíclicas por naturaleza**:
```
arterial → venous → late → [vuelve a] arterial
```

Un modelo softmax tradicional trata las clases como independientes:
- No entiende que "late" está más cerca de "arterial" que de "venous"
- No proporciona medida de confianza natural
- No permite regresión continua dentro de cada fase

### Solución: Representación Circular

Mapeamos las 3 fases a **ángulos equidistantes** en el plano complejo:

```
arterial = 0°   (0 radianes)
venous   = 120° (2π/3 radianes)
late     = 240° (4π/3 radianes)
```

El modelo predice coordenadas **(x, y)** sin normalizar:
- **Ángulo** θ = atan2(y, x) → fase predicha
- **Radio** r = √(x² + y²) → confianza del modelo

---

## 🛠️ Instalación

Ya está integrado en tu pipeline. Solo necesitas verificar los imports:

```bash
cd "c:\Users\juand\Desktop\BIOMED DTU\4. THESIS\kidneys-ct-phase"
python scripts/test_circular_model.py
```

Si el test pasa, estás listo para entrenar.

---

## 🚀 Quick Start

### 1⃣ Test rápido

```bash
python scripts/test_circular_model.py
```

Esto verifica:
- Forward pass
- Angular loss
- Conversión ángulo ↔ fase
- Cálculo de confianza

### 2⃣ Entrenar modelo CNN ligero

```bash
python scripts/train_circular_model.py \
    --architecture cnn \
    --epochs 30 \
    --batch-size 16 \
    --tensors-root outputs_x3/tensors_15ch
```

### 3⃣ Entrenar modelo ResNet (más potente)

```bash
python scripts/train_circular_model.py \
    --architecture resnet \
    --epochs 50 \
    --batch-size 16 \
    --tensors-root outputs_x3/tensors_15ch \
    --lr 1e-4
```

### 4⃣ Evaluar modelo entrenado

```bash
python src/evaluate_circular_model.py \
    --model-path models/circular_phase/best_circular_model.pt \
    --architecture resnet \
    --save-predictions
```

---

## 🧠 Arquitectura del Modelo

### Opción 1: CNN Ligero

```python
CircularPhaseNetCNN(
    in_channels=15  # 5 slices × 3 fases
)
```

Arquitectura:
```
Conv 15→32 + BN + ReLU + Pool
Conv 32→64 + BN + ReLU + Pool
Conv 64→128 + BN + ReLU + AdaptiveAvgPool
Linear 128→2 (x, y)
```

**Parámetros**: ~150K  
**Ventajas**: Rápido, ligero, fácil de entrenar  
**Desventajas**: Menos capacidad representacional

### Opción 2: ResNet18 Adaptado

```python
CircularPhaseNetResNet(
    in_channels=15,
    pretrained=True  # ImageNet weights
)
```

Adaptaciones:
- Primera conv: 3 → 15 canales (copia pesos pretrained)
- Última FC: 512 → 2 (x, y)

**Parámetros**: ~11M  
**Ventajas**: Muy potente, aprovecha pretrained weights  
**Desventajas**: Más lento, requiere más memoria

---

## 🎓 Entrenamiento

### Comando básico

```bash
python scripts/train_circular_model.py
```

### Opciones importantes

| Opción | Descripción | Default |
|--------|-------------|---------|
| `--architecture` | `cnn` o `resnet` | `resnet` |
| `--epochs` | Número de epochs | `30` |
| `--batch-size` | Tamaño de batch | `16` |
| `--lr` | Learning rate | `1e-4` |
| `--tensors-root` | Carpeta con datos | `outputs_x3/tensors_15ch` |
| `--no-pretrained` | Sin pesos ImageNet (solo ResNet) | `False` |
| `--no-augment` | Sin augmentación | `False` |

### Outputs generados

```
models/circular_phase/
├── best_circular_model.pt      # Mejor modelo (val accuracy)
├── last_circular_model.pt       # Último checkpoint
└── training_history.json        # Historial completo
```

### Métricas durante entrenamiento

El trainer muestra:
- **Loss**: Angular loss (1 - cos θ)
- **Accuracy**: Clasificación discreta (0-1-2)
- **Confidence**: Radio promedio (confianza)

Ejemplo:
```
Train Loss: 0.3421 | Acc: 0.892 | Conf: 0.754
Val   Loss: 0.3012 | Acc: 0.915 | Conf: 0.823
```

---

## 📊 Evaluación

### Comando básico

```bash
python src/evaluate_circular_model.py \
    --model-path models/circular_phase/best_circular_model.pt \
    --architecture resnet
```

### Outputs generados

```
outputs/evaluation_circular/
├── confusion_matrix.png           # Matriz de confusión
├── circular_predictions.png       # Predicciones en plano circular
├── confidence_distribution.png    # Distribución de confianza
├── evaluation_results.json        # Métricas JSON
└── predictions.npz                # Predicciones guardadas (opcional)
```

### Interpretación de resultados

#### 1. Confusion Matrix
Muestra clasificación discreta (arterial/venous/late).

#### 2. Circular Predictions Plot
- **Puntos**: Predicciones del modelo (color = fase)
- **Tamaño del punto**: Confianza (radio)
- **Estrellas negras**: Targets ideales (unit circle)
- **Círculo discontinuo**: Unit circle (confianza = 1)

**Interpretación**:
- Puntos grandes cerca de estrellas = buenas predicciones con alta confianza
- Puntos pequeños = predicciones inseguras (útil para detección de ambigüedad)

#### 3. Confidence Distribution
Histograma de confianza por fase.

**Interpretación**:
- Distribución centrada en 0.8-1.0 = modelo muy confiado
- Distribución centrada en 0.3-0.5 = modelo inseguro
- Distribución bimodal = distingue bien casos fáciles vs difíciles

---

## 🆚 Ventajas vs Softmax

| Aspecto | Softmax | Circular |
|---------|---------|----------|
| **Estructura cíclica** | ❌ No respeta | ✅ Respeta naturalmente |
| **Confianza** | ⚠️ Calibración compleja | ✅ Radio = confianza directa |
| **Predicciones ambiguas** | ❌ Difícil detectar | ✅ Radio bajo = ambigüedad |
| **Regresión continua** | ❌ Solo discreto | ✅ Posible within-phase |
| **Entrenamiento** | ✅ Más estable | ⚠️ Requiere más tuning |
| **Interpretabilidad** | ⚠️ Logits abstractos | ✅ Visualizable en 2D |

### Ejemplo práctico

Imagina un caso **entre arterial y venous**:

**Softmax**:
```
P(arterial) = 0.48
P(venous)   = 0.47
P(late)     = 0.05
```
→ Predice "arterial", pero con incertidumbre no cuantificable.

**Circular**:
```
x = 0.35, y = 0.28
radius = 0.45  (baja confianza!)
angle = 38°    (entre 0° y 120°)
```
→ Predice "arterial", pero **r=0.45 indica ambigüedad**.

---

## 🔬 Casos de Uso Avanzados

### 1. Filtrar predicciones inciertas

```python
xy, radius, angle = model(images)

# Solo usar predicciones con alta confianza
confident_mask = radius > 0.7
confident_angles = angle[confident_mask]
confident_labels = labels[confident_mask]
```

### 2. Regresión continua "within-phase"

```python
# Ángulo predicho dentro de cada fase
def get_within_phase_progress(angle, phase):
    phase_center = PHASE_TO_ANGLE[phase]
    deviation = angle - phase_center
    
    # Normalizar a [-1, 1] dentro de sector de 120°
    progress = deviation / (np.pi / 3)
    return progress

# Ejemplo: 
# - progress = -0.5 → inicio de fase
# - progress =  0.0 → centro de fase
# - progress = +0.5 → final de fase
```

### 3. Detección de anomalías

```python
# Casos con baja confianza pueden ser ambiguos o anómalos
anomaly_mask = radius < 0.3

# Revisar manualmente
for idx in torch.where(anomaly_mask)[0]:
    print(f"Case {idx}: low confidence {radius[idx]:.3f}")
```

### 4. Ensemble con modelo discreto

```python
# Combinar predicciones de modelo circular y softmax
circular_pred = angle_to_phase(angle)
softmax_pred = softmax_logits.argmax(dim=1)

# Usar circular solo si confianza alta
final_pred = torch.where(
    radius > 0.7,
    circular_pred,
    softmax_pred
)
```

---

## 🔥 Tips de Entrenamiento

### 1. Learning Rate

- **ResNet pretrained**: `1e-4` (default)
- **CNN from scratch**: `1e-3` (más alto)
- **Fine-tuning**: `1e-5` (más bajo)

### 2. Batch Size

- **GPU pequeña (4-6 GB)**: batch_size=8
- **GPU mediana (8-12 GB)**: batch_size=16
- **GPU grande (>16 GB)**: batch_size=32

### 3. Augmentation

Dejar activado por defecto. Incluye:
- Random horizontal/vertical flip
- Random rotation (pequeña)
- Elastic deformation
- Intensity scaling

### 4. Overfitting

Si val_loss sube pero train_loss baja:
- Aumentar `--weight-decay` (default: 1e-5 → 1e-4)
- Activar dropout (requiere modificar arquitectura)
- Más augmentación

### 5. Underfitting

Si ambas losses se quedan altas:
- Usar `--architecture resnet` en lugar de `cnn`
- Aumentar `--epochs`
- Reducir `--weight-decay`

---

## 📈 Métricas Esperadas

### CNN ligero (30 epochs)
- **Train accuracy**: 85-90%
- **Val accuracy**: 80-85%
- **Mean confidence**: 0.6-0.7

### ResNet pretrained (50 epochs)
- **Train accuracy**: 92-96%
- **Val accuracy**: 88-92%
- **Mean confidence**: 0.75-0.85

### 🎯 Objetivo clínico
- **Val accuracy**: >90% (excelente)
- **Mean confidence**: >0.8 (muy confiable)

---

## 🐛 Troubleshooting

### Problema: Training loss no baja

**Posibles causas**:
1. Learning rate muy bajo → aumentar a 1e-3
2. Gradientes exploding → reducir LR a 1e-5
3. Datos mal cargados → verificar con `test_circular_model.py`

### Problema: Accuracy estancada en ~33%

**Causa**: El modelo predice siempre la misma clase.

**Solución**:
1. Verificar balance de clases en dataset
2. Reducir learning rate
3. Usar pretrained weights (ResNet)

### Problema: Confianza muy baja (<0.3)

**Causa**: El modelo es muy incierto.

**Solución**:
1. Entrenar más epochs
2. Reducir weight_decay
3. Usar arquitectura más potente (ResNet)

### Problema: Loss negativo o NaN

**Causa**: Problema con gradientes.

**Solución**:
1. Reducir learning rate drásticamente
2. Verificar que angular_loss esté bien implementado
3. Clipear gradientes: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`

---

## 📚 Referencias

- **Angular Loss**: Basado en cosine embedding loss
- **Circular Statistics**: Fisher distribution, von Mises
- **Uncertainty**: Predictive uncertainty via model confidence

---

## 🎯 Next Steps

### Mejoras posibles:

1. **Heteroscedastic Uncertainty**: Predicción explícita de varianza
   ```python
   # Predecir (x, y, log_var)
   head = nn.Linear(128, 3)
   ```

2. **Within-Phase Regression**: Continuo dentro de cada fase
   ```python
   # Predecir (phase_logits, within_phase_continuous)
   ```

3. **Multi-task Learning**: Fase global + regresión fina
   ```python
   # Dos heads: circular + discreto
   ```

4. **Attention Mechanism**: Para seleccionar slices más informativos
   ```python
   # Spatial + channel attention en features
   ```

---

## ✅ Checklist de Uso

- [ ] Test rápido pasado (`test_circular_model.py`)
- [ ] Datos preparados en `tensors_15ch/`
- [ ] GPU disponible (recomendado)
- [ ] Entrenar modelo CNN o ResNet
- [ ] Evaluar en validación
- [ ] Visualizar predicciones circulares
- [ ] Comparar con modelo Softmax tradicional
- [ ] Analizar casos de baja confianza
- [ ] Guardar mejor modelo para producción

---

**¡Listo para entrenar!** 🚀

```bash
python scripts/train_circular_model.py --architecture resnet --epochs 50
```
