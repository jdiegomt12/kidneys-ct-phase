# Deformaciones 3D - Guía de Implementación

## Overview

Sistema de augmentación 3D mediante campos de deformación TPS (Thin Plate Spline) aplicados a subvolúmenes de riñones. 

**Características principales:**
- Pre-genera N campos de deformación reproducibles
- Aplica la **misma deformación** a las 3 fases (arterial, venous, late) → garantiza correspondencia
- Deforma solo la región del riñón (z_min a z_max) → preserva el resto del volumen
- Blending suave en los bordes → evita artefactos

## Componentes

### 1. `augmentation.py` - Funciones 3D
**Nuevas funciones:**

```python
from augmentation import (
    generate_deformation_field_3d,    # Genera campo aleatorio
    apply_deformation_field_3d,       # Aplica campo a volumen
    generate_and_save_deformation_fields,  # Pre-genera N campos
    load_deformation_field,           # Carga campo guardado
)
```

**Detalles técnicos:**
- Generador: Grid de puntos de control 4×4×4 con desplazamientos aleatorios
- Interpolación: RegularGridInterpolator (trilineal)
- Suavizado: Gaussian filter (σ=1.5) para suavidad
- Orden de interpolación: lineal (más rápido) o cúbico (mejor calidad)

### 2. `deformation_3d.py` - Orquestación
**Función principal:**

```python
from deformation_3d import (
    deform_kidney_volume,        # Deforma 1 caso/fase
    deform_case_all_phases,      # Deforma todas fases igual
    load_features_csv,           # Lee limites z_min/z_max
)

# Usar:
volume_deformed = deform_kidney_volume(
    volume=vol_zyx,              # (Z, H, W)
    case_id='123',
    phase='arterial',
    features_dict=features,      # {'case_123_arterial': {'z_min': 20, 'z_max': 80}}
    deformation_fields_dir=Path("outputs/deform_fields"),
    field_idx=0,                 # Usar campo #0
    padding=5,                   # Extra padding alrededor del riñón
    apply_blend=True,            # Suavizar transiciones
)
```

**¿Qué hace?**
1. Lee z_min y z_max del riñón desde CSV de features
2. Extrae subvolumen con padding
3. Aplica campo de deformación TPS al subvolumen
4. Inserta subvolumen deformado en volumen original
5. Aplica blending suave en los bordes
6. Retorna volumen completo con región deformada

### 3. `augment_3d_deformation.py` - Batch processing
Script que procesa todos los casos:

```bash
python src/augment_3d_deformation.py \
  --tensors-root outputs_x3/tensors_15ch \
  --features-csv outputs_x3/kidney_features.csv \
  --num-deformations 3 \
  --max-displacement 3.0
```

**Output:**
```
outputs_x3/tensors_15ch_3d_deformed/
  case_1/
    case_1_arterial_def0.npy
    case_1_arterial_def1.npy
    case_1_arterial_def2.npy
    case_1_venous_def0.npy
    ...
```

## Flujo de trabajo

### Paso 1: Generar campos de deformación
```bash
python scripts/augment_3d.py --num-deformations 3
```

Output: `outputs/deform_fields/`
```
outputs/deform_fields/
  deform_field_00.npy   (3, 512, 512, 512) - desplazamientos (dz, dy, dx)
  deform_field_01.npy
  deform_field_02.npy
  metadata.json
```

**Solo se hace una vez** - Los campos son reproducibles (seed fijo = 42+id)

### Paso 2: Aplicar deformaciones a tensores
```bash
python src/augment_3d_deformation.py \
  --tensors-root outputs_x3/tensors_15ch \
  --features-csv outputs_x3/kidney_features.csv \
  --num-deformations 3
```

**Output:** Tensores deformados en `outputs_x3/tensors_15ch_3d_deformed/`

## Parámetros importantes

| Parámetro | Default | Rango recomendado | Notas |
|-----------|---------|-------------------|-------|
| `max_displacement` | 3.0 voxels | 2–5 | Más → más variación pero menos realismo |
| `num_control_points` | 4×4×4 | 3–5 | Más → más variaciones locales |
| `padding` | 5 voxels | 3–10 | Buffer alrededor de z_min/z_max |
| `blend_width` | 3 voxels | 2–5 | Transición suave en bordes |

## Cómo funcionan las deformaciones

### Grid de puntos de control
```
Volume: (100, 256, 256)
Control points: 4×4×4 = 64 puntos
```

Distribuidos uniformemente en el volumen. Cada punto tiene un desplazamiento aleatorio `[-max_disp, +max_disp]`.

### Interpolación
```python
displacement[z, y, x] = interp(control_points, displacements)
```

Interpolación trilineal entre puntos de control → suave.

### Suavizado
```python
displacement = gaussian_filter(displacement, sigma=1.5)
```

Elimina discontinuidades, produce deformaciones naturales.

### Aplicación
```python
new_coords = old_coords + displacement
new_image[z, y, x] = old_image[z + dz, y + dy, x + dx]
```

Map_coordinates con interpolación orden-1 (lineal).

## Integración en pipeline actual

### Opción A: Post-processing (recomendado)
```python
# En main.py, antes de borrar vol:
from deformation_3d import deform_kidney_volume

vol_def = deform_kidney_volume(
    volume=vol.hu_zyx,
    case_id=entry['case_id'],
    phase=entry['phase'],
    features_dict=features_dict,
    field_idx=epoch,  # Diferentes campos en iteraciones sucesivas
)

# Guardar vol_def en otra carpeta
```

### Opción B: Script offline (más limpio)
```bash
# Después de que main.py termine:
python src/augment_3d_deformation.py \
  --tensors-root outputs_x3/tensors_15ch \
  --features-csv outputs_x3/kidney_features.csv
```

**Recomendación:** Opción B
- No ralentiza el pipeline base
- Campos reproducibles
- Fácil de debuggear
- Control fino sobre qué deformaciones aplicar

## Troubleshooting

### "deform_field_00.npy not found"
```bash
python scripts/augment_3d.py
```

Generar primero los campos.

### Output tiene artefactos en bordes
- Aumentar `padding` (5 → 8)
- Aumentar `blend_width` (3 → 5)

### Deformación demasiado fuerte
- Bajar `max_displacement` (3.0 → 2.0)
- Reducir `num_control_points` (4 → 3)

### Demasiado lento
- Usar orden interpolación 1 (lineal) en lugar de 3 (cúbico)
- Reducir resolución si es posible

## Validación

```python
import numpy as np

# Verificar que deformación es razonable
diff = np.abs(vol_original - vol_deformed).mean()
print(f"Cambio medio: {diff:.4f}")  # Debería ser pequeño (< 0.1 típicamente)

# Verificar que se deformó solo la región correcta
z_min, z_max = 20, 80
outside = np.abs(vol_original[:z_min, :, :] - vol_deformed[:z_min, :, :]).mean()
inside = np.abs(vol_original[z_min:z_max, :, :] - vol_deformed[z_min:z_max, :, :]).mean()

print(f"Cambio fuera region: {outside:.6f}")  # Debería ser ~0
print(f"Cambio dentro region: {inside:.4f}")  # Debería ser > outside
```

## Performance

Con GPU (si se implementa):
- Generar 3 campos: ~1 segundo
- Aplicar 1 campo a volumen (512³): ~0.5 segundos

Con CPU:
- Generar 3 campos: ~2 segundos
- Aplicar 1 campo a volumen: ~1-2 segundos

Para 100 casos × 3 deformaciones = 5-10 minutos aprox.

## Próximas mejoras

1. **GPU acceleration** - Usar PyTorch/CuPy para aplicar campos
2. **Elastic regularization** - Limitar deformación local para más realismo
3. **Anatomical constraints** - Respetar límites del riñón (máscara de segmentación)
4. **Multi-resolution** - Coarse-to-fine para mejor rendimiento
5. **Diferentiable** - Integrar en training del modelo DL como augmentación on-the-fly
