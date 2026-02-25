"""
Dataset para cargar tensores multifase desde carpeta tensors_15ch.

Estructura esperada:
  tensors_15ch/
    case_1/
      case_1_arterial.npy
      case_1_venous.npy
      case_1_late.npy
    case_2/
      ...
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data_augmentation.augmentation import get_augmentation_pipeline, augment_tensor_multimodal


PHASE_NAMES = ['arterial', 'venous', 'late']
PHASE_TO_LABEL = {phase: idx for idx, phase in enumerate(PHASE_NAMES)}


class PhaseDataset(Dataset):
    """
    Dataset que carga 3 tensores (fases) y retorna uno aleatorio como target.
    
    Estrategia: Para cada caso, tenemos 3 fases. Seleccionamos una como target
    y usamos las otras 2 como features que necesitan ser clasificadas.
    """
    
    def __init__(
        self,
        tensors_root: Path,
        transform=None,
        return_all_phases: bool = False,
        normalize: bool = True,
    ):
        """
        Args:
            tensors_root: ruta a carpeta tensors_15ch
            transform: pipeline de augmentación (albumentations)
            return_all_phases: si True, retorna tensor con 3 fases concatenadas
                              si False, retorna una fase aleatoria
            normalize: normalizar a [0, 1] o [-1, 1]
        """
        self.tensors_root = Path(tensors_root)
        self.transform = transform
        self.return_all_phases = return_all_phases
        self.normalize = normalize
        
        # Encontrar todos los casos
        self.cases = self._find_cases()
        
        if not self.cases:
            raise ValueError(f"No se encontraron casos en {tensors_root}")
    
    def _find_cases(self) -> List[Tuple[str, Path]]:
        """Encuenta todos los casos con sus 3 fases."""
        cases = []
        
        for case_folder in sorted(self.tensors_root.iterdir()):
            if not case_folder.is_dir():
                continue
            
            case_id = case_folder.name
            
            # Verificar que existan los 3 archivos
            phase_files = {phase: case_folder / f"{case_id}_{phase}.npy"
                          for phase in PHASE_NAMES}
            
            if all(f.exists() for f in phase_files.values()):
                cases.append((case_id, case_folder))
        
        print(f"[INFO] Dataset: encontrados {len(cases)} casos")
        return cases
    
    def _load_case(self, case_folder: Path, case_id: str) -> Dict[str, np.ndarray]:
        """Carga los 3 tensores de un caso."""
        tensors = {}
        for phase in PHASE_NAMES:
            path = case_folder / f"{case_id}_{phase}.npy"
            tensors[phase] = np.load(path)
        return tensors
    
    def __len__(self) -> int:
        if self.return_all_phases:
            # 1 sample por caso (3 fases apiladas)
            return len(self.cases)
        else:
            # 3 samples por caso (1 fase como label)
            return len(self.cases) * 3
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            {
                'image': tensor (15, 512, 512) - una o tres fases
                'label': int - índice de la fase (0=arterial, 1=venous, 2=late)
                'case_id': str
            }
        """
        if self.return_all_phases:
            case_idx = idx
            case_id, case_folder = self.cases[case_idx]
            
            # Cargar todas las fases
            tensors = self._load_case(case_folder, case_id)
            
            # Aplicar augmentación si disponible
            if self.transform is not None:
                tensors = augment_tensor_multimodal(tensors, self.transform)
            
            # Apilar: (15, 512, 512) por fase → (15, 512, 512)
            # Las 5 primeras son arterial, next 5 venous, last 5 late
            image = np.concatenate(
                [tensors[phase] for phase in PHASE_NAMES],
                axis=0
            )  # (45, 512, 512)
            
            # Para este modo, label es dummy (no se usa)
            label = 0
            target_phase = "all"
        
        else:
            # Modo: 3 samples por caso
            case_idx = idx // 3
            phase_idx = idx % 3
            
            case_id, case_folder = self.cases[case_idx]
            target_phase = PHASE_NAMES[phase_idx]
            
            # Cargar todas las fases
            tensors = self._load_case(case_folder, case_id)
            
            # Aplicar augmentación si disponible
            if self.transform is not None:
                tensors = augment_tensor_multimodal(tensors, self.transform)
            
            # Retornar solo la fase target
            image = tensors[target_phase]  # (15, 512, 512)
            label = phase_idx
        
        # Normalizar
        if self.normalize:
            image = image.astype(np.float32)
            image = np.clip(image, 0, 1)  # Asegurar [0, 1]
        
        # A tensor
        image_tensor = torch.from_numpy(image).float()
        
        return {
            'image': image_tensor,
            'label': label,
            'case_id': case_id,
            'phase': target_phase,
        }


def create_dataloaders(
    tensors_root: Path,
    batch_size: int = 16,
    num_workers: int = 0,
    train_split: float = 0.8,
    augment_train: bool = True,
    return_all_phases: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function para crear train/val dataloaders.
    
    Args:
        tensors_root: ruta a tensors_15ch
        batch_size: batch size
        num_workers: workers para data loading
        train_split: proporción train/val
        augment_train: aplicar augmentación a train (no a val)
        return_all_phases: si True, retorna 3 fases concatenadas
    
    Returns:
        (train_loader, val_loader)
    """
    tensors_root = Path(tensors_root)
    
    # Crear dataset sin augmentación primero para obtener casos
    dataset = PhaseDataset(
        tensors_root=tensors_root,
        transform=None,
        return_all_phases=return_all_phases,
        normalize=True,
    )
    
    # Split train/val
    n_cases = len(dataset.cases)
    split_idx = int(n_cases * train_split)
    
    train_cases = dataset.cases[:split_idx]
    val_cases = dataset.cases[split_idx:]
    
    print(f"[INFO] Train cases: {len(train_cases)}, Val cases: {len(val_cases)}")
    
    # Crear datasets con/sin augmentación
    transform_train = get_augmentation_pipeline() if augment_train else None
    
    dataset_train = PhaseDataset(
        tensors_root=tensors_root,
        transform=transform_train,
        return_all_phases=return_all_phases,
        normalize=True,
    )
    # Limitar a train cases
    dataset_train.cases = train_cases
    
    dataset_val = PhaseDataset(
        tensors_root=tensors_root,
        transform=None,
        return_all_phases=return_all_phases,
        normalize=True,
    )
    # Limitar a val cases
    dataset_val.cases = val_cases
    
    # Dataloaders
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader
