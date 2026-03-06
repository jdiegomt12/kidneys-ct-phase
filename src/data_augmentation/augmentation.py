import albumentations as A
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
import json
from scipy.ndimage import map_coordinates, gaussian_filter


def get_augmentation_pipeline():
    """Pipeline de augmentación para múltiples fases simultáneamente"""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=0,
                p=0.5,
            ),
            A.GaussNoise(var_limit=(0.0005, 0.002), p=0.3),
        ],
        additional_targets={
            'image1': 'image',  # phase 2 (venous)
            'image2': 'image',  # phase 3 (late)
        }
    )


def augment_tensor(tensor: np.ndarray, transform):
    """
    Augmentación single-phase (legacy).
    tensor: (15, H, W) → (15, H, W)
    """
    tensor_hw_c = np.transpose(tensor, (1, 2, 0))
    augmented = transform(image=tensor_hw_c)
    aug_hw_c = augmented["image"]
    aug_c_hw = np.transpose(aug_hw_c, (2, 0, 1))
    return aug_c_hw.astype(np.float32)


def augment_tensor_multimodal(tensors_dict: Dict[str, np.ndarray], transform) -> Dict[str, np.ndarray]:
    """
    Augmentación sincronizada para múltiples fases.
    La MISMA transformación se aplica a todas las fases.
    
    Args:
        tensors_dict: {'arterial': (15,H,W), 'venous': (15,H,W), 'late': (15,H,W)}
        transform: A.Compose pipeline
    
    Returns:
        Dict con las mismas claves pero con tensores augmentados
    """
    # Transponer todos a (H, W, C)
    phases = list(tensors_dict.keys())
    transposed = {phase: np.transpose(tensors_dict[phase], (1, 2, 0))
                  for phase in phases}
    
    # Aplicar la MISMA transformación a todas las fases
    aug_data = {'image': transposed[phases[0]]}
    if len(phases) > 1:
        aug_data['image1'] = transposed[phases[1]]
    if len(phases) > 2:
        aug_data['image2'] = transposed[phases[2]]
    
    augmented = transform(**aug_data)
    
    # Volver a (C, H, W) y construir resultado
    result = {}
    for i, phase in enumerate(phases):
        key = 'image' if i == 0 else f'image{i}'
        result[phase] = np.transpose(augmented[key], (2, 0, 1)).astype(np.float32)
    
    return result


# ============================================================================
# DEFORMACIONES 3D - TPS (Thin Plate Spline)
# ============================================================================

def generate_deformation_field_3d(
    shape: Tuple[int, int, int],
    num_control_points: int = 4,
    max_displacement: float = 3.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Genera un campo de deformación 3D de forma eficiente en memoria.
    
    Args:
        shape: (D, H, W) - dimensiones del volumen
        num_control_points: número de puntos de control por dimensión
        max_displacement: desplazamiento máximo en voxels
        seed: seed para reproducibilidad
    
    Returns:
        displacement_field: (3, D, H, W) - desplazamientos en x, y, z
    """
    if seed is not None:
        np.random.seed(seed)
    
    D, H, W = shape
    
    # Crear malla de puntos de control uniformemente espaciados
    z_ctrl = np.linspace(0, D - 1, num_control_points)
    y_ctrl = np.linspace(0, H - 1, num_control_points)
    x_ctrl = np.linspace(0, W - 1, num_control_points)
    
    # Generar desplazamientos aleatorios pequeños en cada punto de control
    displacements_z = np.random.uniform(-max_displacement, max_displacement, 
                                        (num_control_points, num_control_points, num_control_points))
    displacements_y = np.random.uniform(-max_displacement, max_displacement,
                                        (num_control_points, num_control_points, num_control_points))
    displacements_x = np.random.uniform(-max_displacement, max_displacement,
                                        (num_control_points, num_control_points, num_control_points))
    
    # Suavizar los desplazamientos en los puntos de control
    displacements_z = gaussian_filter(displacements_z, sigma=0.5)
    displacements_y = gaussian_filter(displacements_y, sigma=0.5)
    displacements_x = gaussian_filter(displacements_x, sigma=0.5)
    
    # Procesamiento por slices para ahorrar memoria
    from scipy.interpolate import RegularGridInterpolator
    
    disp_z_full = np.zeros((D, H, W), dtype=np.float32)
    disp_y_full = np.zeros((D, H, W), dtype=np.float32)
    disp_x_full = np.zeros((D, H, W), dtype=np.float32)
    
    # Interpolar por slices Z
    chunk_size = 50  # Procesar de 50 en 50 slices
    
    for z_start in range(0, D, chunk_size):
        z_end = min(z_start + chunk_size, D)
        
        # Crear grid para este chunk
        grid_z, grid_y, grid_x = np.meshgrid(
            np.arange(z_start, z_end),
            np.arange(H),
            np.arange(W),
            indexing='ij'
        )
        
        # Crear interpoladores
        interp_z = RegularGridInterpolator(
            (z_ctrl, y_ctrl, x_ctrl), displacements_z,
            bounds_error=False, fill_value=0.0, method='linear'
        )
        interp_y = RegularGridInterpolator(
            (z_ctrl, y_ctrl, x_ctrl), displacements_y,
            bounds_error=False, fill_value=0.0, method='linear'
        )
        interp_x = RegularGridInterpolator(
            (z_ctrl, y_ctrl, x_ctrl), displacements_x,
            bounds_error=False, fill_value=0.0, method='linear'
        )
        
        # Stack points
        chunk_len = (z_end - z_start) * H * W
        points = np.stack([grid_z.ravel(), grid_y.ravel(), grid_x.ravel()], axis=1)
        
        # Interpolar
        disp_z_full[z_start:z_end, :, :] = interp_z(points).reshape((z_end - z_start, H, W))
        disp_y_full[z_start:z_end, :, :] = interp_y(points).reshape((z_end - z_start, H, W))
        disp_x_full[z_start:z_end, :, :] = interp_x(points).reshape((z_end - z_start, H, W))
    
    # Suavizar resultado final
    sigma = 1.0
    disp_z_full = gaussian_filter(disp_z_full, sigma=sigma)
    disp_y_full = gaussian_filter(disp_y_full, sigma=sigma)
    disp_x_full = gaussian_filter(disp_x_full, sigma=sigma)
    
    # Stack en (3, D, H, W)
    displacement_field = np.stack([disp_z_full, disp_y_full, disp_x_full], axis=0)
    
    return displacement_field.astype(np.float32)


def apply_deformation_field_3d(
    volume: np.ndarray,
    displacement_field: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """
    Aplica un campo de deformación 3D a un volumen.
    
    Args:
        volume: (D, H, W) - volumen
        displacement_field: (3, D, H, W) - desplazamientos (dz, dy, dx)
        order: orden de interpolación (1=lineal, 3=cúbico)
    
    Returns:
        deformed_volume: (D, H, W)
    """
    D, H, W = volume.shape
    
    # Crear coordenadas de muestreo
    z_coords, y_coords, x_coords = np.meshgrid(
        np.arange(D), np.arange(H), np.arange(W),
        indexing='ij'
    )
    
    # Aplicar desplazamientos
    z_new = z_coords + displacement_field[0]
    y_new = y_coords + displacement_field[1]
    x_new = x_coords + displacement_field[2]
    
    # Clip a límites válidos
    z_new = np.clip(z_new, 0, D - 1)
    y_new = np.clip(y_new, 0, H - 1)
    x_new = np.clip(x_new, 0, W - 1)
    
    # Interpolar
    coords = np.array([z_new, y_new, x_new])
    deformed = map_coordinates(volume, coords, order=order, cval=0.0)
    
    return deformed.astype(volume.dtype)


def save_deformation_field(
    displacement_field: np.ndarray,
    output_path: Path,
    field_id: int,
) -> Path:
    """Guarda un campo de deformación."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    field_path = output_path / f"deform_field_{field_id:02d}.npy"
    np.save(field_path, displacement_field)
    
    return field_path


def load_deformation_field(
    field_path: Path,
) -> np.ndarray:
    """Carga un campo de deformación."""
    return np.load(field_path)


def generate_and_save_deformation_fields(
    volume_shape: Tuple[int, int, int],
    num_fields: int = 3,
    output_dir: Path = Path("outputs/deform_fields"),
    max_displacement: float = 3.0,
) -> list:
    """
    Genera y guarda múltiples campos de deformación.
    
    Args:
        volume_shape: (D, H, W)
        num_fields: número de campos a generar
        output_dir: dónde guardar
        max_displacement: desplazamiento máximo
    
    Returns:
        Lista de rutas de campos guardados
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for field_id in range(num_fields):
        # Generar con seed fijo para reproducibilidad
        deform_field = generate_deformation_field_3d(
            shape=volume_shape,
            num_control_points=4,
            max_displacement=max_displacement,
            seed=42 + field_id,
        )
        
        # Guardar
        path = save_deformation_field(deform_field, output_dir, field_id)
        saved_paths.append(path)
        print(f"[*] Campo {field_id + 1}/{num_fields} guardado: {path}")
    
    # Guardar metadata
    metadata = {
        'num_fields': num_fields,
        'volume_shape': list(volume_shape),
        'max_displacement': max_displacement,
        'fields': [str(p) for p in saved_paths],
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[*] Metadata guardado: {metadata_path}")
    
    return saved_paths
