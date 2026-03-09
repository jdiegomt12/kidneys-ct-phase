# Errores Comunes de Indexing y Soluciones

Basado en el primer test con 45 casos, aquí están los **3 errores principales** detectados:

## 🔴 Error 1: Temporal Order Inversión (~42% de casos)

**Problema:** El orden de adquisición no sigue arterial < venous < late

```
arterial=140201, venous=140048, late=135946  ❌ Orden incorrecto
Expected: arterial < venous < late
```

**Causas posibles:**
- Los datos reales tienen tiempos grabados incorrectamente
- El campo AcquisitionTime puede ser incorrecto en algunos DICOM
- Posible problema de sincronización en el scanner

**Solución propuesta:**
- Usar `SliceLocation` (coordenada Z) en lugar de tiempo como criterio de orden
- Arterial typically captura slices más altos (supeficie renal inicial)
- Venous y late escaneandose regiones más profundas
- Usar diferencia espacial en lugar de temporal

---

## 🔴 Error 2: Múltiples Candidatos Para Fase (~18% de casos)

**Problema:** 2 series compiten por la misma fase

```
⚠️ Multiple candidates for arterial: found 2
  1. 42210000: Files=650, Slices=1, Time=084226...
  2. 39370000: Files=590, Slices=5, Time=083942...
```

**Causas posibles:**
- Scanner grabó 2 acquisiciones del mismo phase (rescans)
- Directorios con estructura inconsistente
- Mixed datasets de diferentes protocolos

**Solución aplicada:**
- Si diferencia en archivos > 100 → auto-resolver por cantidad
- Si diferencia < 100 → enviar a manual

**Mejora futura:**
- Usar SliceThickness como criterio adicional
- Comparar spacing voxel
- Validate image dimensions consistency

---

## 🔴 Error 3: No Phase Match (~1-5% de casos)

**Problema:** Carpeta con DICOMs válidos pero sin tokens reconocibles

```
❌ NO_PHASE: folder name "33490000" no contiene ningún token
En PHASE_TOKENS: ["arterial", "artery", "aortic", "aorta", "art"]
                 ["venous", "portal", "pv", "portovenous", "vena"]
                 ["late", "delayed", "delay", "excretory", "urographic"]
```

**Causas posibles:**
- Datos de fuente diferente con naming diferente
- Carpetas con solo números sin metadata en nombre
- Series Description tiene nombres no estándar

**Solución propuesta:**
- Leer `SeriesDescription` del metadata DICOM
- Buscar tokens dentro de SeriesDescription además del path
- Usar ML/fuzzy matching para nombres no exactos
- Fallback a orden temporal si todo lo demás falla

---

## 📊 Estadísticas del Test

```
Total casos:         45
Exitosos:           37 (82%)
Fallidos:            8 (18%)

Temporal issues:    ~19 casos (pero CONTINUARON exitosos)
Multiple candidates: ~8 casos (FALLARON, fueron a manual)
NO_PHASE:           ~1-2 casos (embedded en failures)
```

---

## ✅ Recomendaciones Implementadas

### 1. **Prints Mejorados** 
Ahora muestra por cada candidato:
```
- folder_name: phase | files=600, slices=234, time=120050 {LOW_FILES, NO_TOKEN_MATCH}
```

### 2. **Detección de Anomalías**
- `⚠️ SUSPICIOUS_SLICES(1)` → probablemente archivo corrupto
- `NO_TOKEN_MATCH` → necesita lógica adicional
- `LOW_FILES(45)` → posible series incompleta

### 3. **Auto-resolución inteligente**
- Si 2 candidatos y diferencia > 100 archivos → pick automáticamente
- Sino → manual resolution

---

## 🚀 Próximos Pasos

1. **Usar SeriesDescription** además de folder name para phase detection
2. **Implementar SliceLocation ordering** en lugar de solo tiempo
3. **Validate slice count** (200-300 normal, <10 = sospechoso)
4. **Progressive confidence scoring** en lugar de binario
5. **Store why each was chosen** en CSV para audit

