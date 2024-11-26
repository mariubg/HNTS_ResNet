import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json
import logging
from typing import Optional, Dict

# Import the data loading functions from previous artifact
from preprocessing import create_dataloaders
from ResNet.ResNetUNet3D import ResNetUNet3D

def setup_logging(output_dir: Path) -> None:
    """Configure logging to both file and console"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = torch.sigmoid(predictions)
        
        # Flatten predictions and targets
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        return 1 - dice

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: Path,
        learning_rate: float = 1e-4,
        save_interval: int = 5
    ):
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.save_interval = save_interval
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup loss and optimizer
        self.criterion = DiceLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Setup logging
        setup_logging(self.output_dir)
        
        # Initialize best metrics
        self.best_val_loss = float('inf')
        
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        
        # Save regular checkpoint
        if epoch % self.save_interval == 0:
            torch.save(checkpoint, self.output_dir / f'checkpoint_epoch_{epoch}.pt')
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best_model.pt')
            
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc='Training') as pbar:
            for batch_idx, (images, masks) in enumerate(pbar):
                images = images.to(self.device)
                masks = masks.to(self.device)
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)

                loss = self.criterion(outputs, masks)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update progress bar
                total_loss += loss.item()
                pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})
                
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Run validation"""
        self.model.eval()
        total_loss = 0
        dice_scores = []
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Calculate Dice score for monitoring
                pred_masks = (torch.sigmoid(outputs) > 0.5).float()
                dice = 1 - self.criterion(pred_masks, masks).item()
                dice_scores.append(dice)
                
                total_loss += loss.item()
        
        metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_dice': np.mean(dice_scores)
        }
        
        return metrics
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int,
        resume_checkpoint: Optional[str] = None
    ) -> None:
        """Main training loop"""
        start_epoch = 0
        
        # Resume from checkpoint if specified
        if resume_checkpoint:
            checkpoint = torch.load(resume_checkpoint)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['val_loss']
            logging.info(f'Resumed training from epoch {start_epoch}')
        
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            logging.info(f'Starting epoch {epoch + 1}/{num_epochs}')
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics['val_loss']
            
            # Log metrics
            logging.info(
                f'Epoch {epoch + 1}/{num_epochs} - '
                f'Train Loss: {train_loss:.4f} - '
                f'Val Loss: {val_loss:.4f} - '
                f'Val Dice: {val_metrics["val_dice"]:.4f}'
            )
            
            # Save checkpoint if best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch + 1, val_loss, is_best=True)
                logging.info(f'New best model saved with validation loss: {val_loss:.4f}')
            
            # Save regular checkpoint
            self.save_checkpoint(epoch + 1, val_loss)

def main():
    # Configuration
    config = {
        'data_dir': 'data\HNTSMRG24_train',
        'output_dir': f'outputs/run_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'patch_size': (128, 128, 64),
        'patches_per_volume': 4,
        'num_workers': 4,
        'batch_size': 1,
        'num_epochs': 200,
        'learning_rate': 1e-4,
        'train_split': 0.8,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'resume_checkpoint': None  # Set to checkpoint path to resume training
    }
    
    # Save configuration
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        train_split=config['train_split'],
        patch_size=config['patch_size'],
        patches_per_volume=config['patches_per_volume'],
        num_workers=config['num_workers']
    )
    
    # Initialize model (replace with your model)
    # model = YourModel()
    
    # Initialize trainer
    trainer = Trainer(
        model=ResNetUNet3D(in_channels=1, out_channels=1),
        device=torch.device(config['device']),
        output_dir=output_dir,
        learning_rate=config['learning_rate']
    )
    
    # Start training
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        resume_checkpoint=config['resume_checkpoint']
    )

if __name__ == '__main__':
    main()