import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import torch
from torch.utils.data import Dataset
import logging
from scipy.ndimage import zoom

class MRIDataset(Dataset):
    def __init__(self, 
                 patient_data: List[Dict[str, str]],
                 patch_size: Optional[Tuple[int, int, int]] = None,
                 patches_per_volume: int = 4,
                 target_size: Optional[Tuple[int, int, int]] = None):
        """
        Dataset class for 3D MRI scans and their segmentation masks
        
        Args:
            patient_data: List of dictionaries containing patient paths
            patch_size: Tuple of (depth, height, width) for patches. If None, use full images
            patches_per_volume: Number of patches to extract per volume (only used if patch_size is not None)
            target_size: Optional target size for full images (only used if patch_size is None)
        """
        self.patient_data = patient_data
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.use_patches = patch_size is not None
        
        # Determine target size for full images if needed
        if not self.use_patches:
            if target_size is None:
                # Load first image to get dimensions if target_size not provided
                first_img = nib.load(patient_data[0]['mri'])
                self.target_size = first_img.shape
                logging.info(f"Using first image dimensions as target size: {self.target_size}")
            else:
                self.target_size = target_size
                logging.info(f"Using provided target size: {self.target_size}")
        
        # Calculate total length
        if self.use_patches:
            self.total_items = len(patient_data) * patches_per_volume
            logging.info(f"Using patches. Total patches that will be generated: {self.total_items}")
        else:
            self.total_items = len(patient_data)
            logging.info(f"Using full images. Total volumes: {self.total_items}")
    
    def __len__(self) -> int:
        return self.total_items
    
    def get_valid_patch_coords(self, volume_shape: Tuple[int, ...]) -> Tuple[slice, ...]:
        """
        Get random but valid patch coordinates
        """
        slices = []
        for dim, patch_dim in zip(volume_shape, self.patch_size):
            valid_start = max(0, dim - patch_dim)
            start = np.random.randint(0, valid_start + 1)
            slices.append(slice(start, start + min(patch_dim, dim)))
        return tuple(slices)

    def extract_patch(self, volume: np.ndarray, coords: Tuple[slice, ...]) -> np.ndarray:
        """
        Extract a patch and pad if necessary
        """
        patch = volume[coords]
        
        # Check if padding is needed
        if patch.shape != self.patch_size:
            pad_dims = []
            for patch_dim, target_dim in zip(patch.shape, self.patch_size):
                pad_size = max(0, target_dim - patch_dim)
                pad_before = pad_size // 2
                pad_after = pad_size - pad_before
                pad_dims.extend([pad_before, pad_after])
            patch = np.pad(patch, pad_dims, mode='constant', constant_values=0)
            
        return patch

    def resize_volume(self, volume: np.ndarray, is_mask: bool = False) -> np.ndarray:
        """
        Resize a volume to target size
        """
        if volume.shape == self.target_size:
            return volume
            
        factors = [t/c for t, c in zip(self.target_size, volume.shape)]
        order = 0 if is_mask else 1  # nearest neighbor for masks, linear for images
        return zoom(volume, factors, order=order)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_patches:
            # Convert global index to patient index and patch number
            patient_idx = idx // self.patches_per_volume
        else:
            # Each index corresponds to one patient
            patient_idx = idx
            
        patient = self.patient_data[patient_idx]
        
        try:
            # Load image and mask
            image = nib.load(patient['mri']).get_fdata().astype(np.float32)
            mask = nib.load(patient['mask']).get_fdata().astype(np.float32)
            
            if self.use_patches:
                # Extract patches
                coords = self.get_valid_patch_coords(image.shape)
                image = self.extract_patch(image, coords)
                mask = self.extract_patch(mask, coords)
            else:
                # Resize full volumes if needed
                if self.target_size != image.shape:
                    image = self.resize_volume(image, is_mask=False)
                    mask = self.resize_volume(mask, is_mask=True)
            
            # Normalize image (not mask)
            image = np.clip(image, a_min=0, a_max=None)
            if image.max() != 0:
                image = (image - image.min()) / (image.max() - image.min())
            
            # Add channel dimension and convert to tensor
            image = torch.from_numpy(image).unsqueeze(0)
            mask = torch.from_numpy(mask).unsqueeze(0)
            
            return image, mask
            
        except Exception as e:
            logging.error(f"Error processing patient {patient['patient_id']}, index {idx}: {str(e)}")
            raise

def create_dataloaders(
    data_dir: str,
    patch_size: Optional[Tuple[int, int, int]] = None,
    patches_per_volume: int = 4,
    target_size: Optional[Tuple[int, int, int]] = None,
    batch_size: int = 4,
    train_split: float = 0.8,
    num_workers: int = 4
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create training and validation dataloaders
    
    Args:
        data_dir: Path to data directory
        patch_size: If provided, use patches of this size. If None, use full images
        patches_per_volume: Number of patches per volume (only used if patch_size is not None)
        target_size: Target size for full images (only used if patch_size is None)
        batch_size: Batch size
        train_split: Fraction of data to use for training
        num_workers: Number of worker processes for data loading
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Get all patient paths
    data_dir = Path(data_dir)
    patient_paths = []
    
    for patient_dir in sorted(data_dir.iterdir()):
        if patient_dir.is_dir():
            mri_path = patient_dir / "preRT_T2.nii.gz"
            mask_path = patient_dir / "preRT_mask.nii.gz"
            
            if mri_path.exists() and mask_path.exists():
                patient_paths.append({
                    'patient_id': patient_dir.name,
                    'mri': str(mri_path),
                    'mask': str(mask_path)
                })
    
    if not patient_paths:
        raise ValueError(f"No valid patient data found in {data_dir}")
    
    logging.info(f"Found {len(patient_paths)} patients")
    
    # Split into train and validation
    num_train = int(len(patient_paths) * train_split)
    train_data = patient_paths[:num_train]
    val_data = patient_paths[num_train:]
    
    # Create datasets
    train_dataset = MRIDataset(
        train_data,
        patch_size=patch_size,
        patches_per_volume=patches_per_volume,
        target_size=target_size
    )
    val_dataset = MRIDataset(
        val_data,
        patch_size=patch_size,
        patches_per_volume=patches_per_volume,
        target_size=target_size
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader