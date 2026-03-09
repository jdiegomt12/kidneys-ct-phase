"""
Comparación lado a lado del modelo Circular vs Softmax tradicional.

Entrena ambos modelos en los mismos datos y compara:
- Accuracy
- Confianza (solo circular)
- Matriz de confusión
- Métricas por fase

Uso:
    python scripts/compare_models.py --tensors-root outputs_x3/tensors_15ch
    python scripts/compare_models.py --epochs 30 --architecture resnet
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Imports circular model
from circular_phase_model import (
    build_circular_model,
    angular_loss,
    phase_to_unit_vector,
    angle_to_phase,
    PHASE_NAMES,
)

# Imports softmax model
from supervised import build_model
from phase_dataset import create_dataloaders


def train_circular_model(train_loader, val_loader, epochs, device, lr):
    """Entrena modelo circular."""
    print("\n" + "="*60)
    print("🔵 TRAINING CIRCULAR MODEL")
    print("="*60)
    
    model = build_circular_model(architecture="resnet", device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            pred_xy, radius, angle = model(images)
            target_xy = phase_to_unit_vector(labels)
            loss = angular_loss(pred_xy, target_xy)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            pred_phases = angle_to_phase(angle)
            train_correct += (pred_phases == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                pred_xy, radius, angle = model(images)
                target_xy = phase_to_unit_vector(labels)
                loss = angular_loss(pred_xy, target_xy)
                
                val_loss += loss.item() * images.size(0)
                pred_phases = angle_to_phase(angle)
                val_correct += (pred_phases == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
    
    print(f"\n✅ Best Val Accuracy: {best_acc:.3f}")
    return model, history, best_acc


def train_softmax_model(train_loader, val_loader, epochs, device, lr):
    """Entrena modelo softmax tradicional."""
    print("\n" + "="*60)
    print("📊 TRAINING SOFTMAX MODEL")
    print("="*60)
    
    model = build_model(num_classes=3, pretrained=True, device=device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, preds = logits.max(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(images)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = logits.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
    
    print(f"\n✅ Best Val Accuracy: {best_acc:.3f}")
    return model, history, best_acc


def evaluate_circular(model, dataloader, device):
    """Evalúa modelo circular."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_radius = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Circular"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            pred_xy, radius, angle = model(images)
            pred_phases = angle_to_phase(angle)
            
            all_preds.extend(pred_phases.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_radius.extend(radius.cpu().numpy())
    
    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'confidence': np.array(all_radius),
    }


def evaluate_softmax(model, dataloader, device):
    """Evalúa modelo softmax."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Softmax"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            _, preds = logits.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probs': np.array(all_probs),
    }


def plot_comparison(circular_results, softmax_results, output_dir):
    """Genera visualizaciones comparativas."""
    
    # 1. Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Circular
    cm_circular = confusion_matrix(circular_results['labels'], circular_results['predictions'])
    sns.heatmap(cm_circular, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=PHASE_NAMES, yticklabels=PHASE_NAMES)
    axes[0].set_title('Circular Model', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Phase')
    axes[0].set_xlabel('Predicted Phase')
    
    # Softmax
    cm_softmax = confusion_matrix(softmax_results['labels'], softmax_results['predictions'])
    sns.heatmap(cm_softmax, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=PHASE_NAMES, yticklabels=PHASE_NAMES)
    axes[1].set_title('Softmax Model', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Phase')
    axes[1].set_xlabel('Predicted Phase')
    
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Accuracy por fase
    fig, ax = plt.subplots(figsize=(10, 6))
    
    phases = range(3)
    circular_accs = []
    softmax_accs = []
    
    for phase in phases:
        mask = circular_results['labels'] == phase
        circular_acc = (circular_results['predictions'][mask] == circular_results['labels'][mask]).mean()
        softmax_acc = (softmax_results['predictions'][mask] == softmax_results['labels'][mask]).mean()
        circular_accs.append(circular_acc)
        softmax_accs.append(softmax_acc)
    
    x = np.arange(len(PHASE_NAMES))
    width = 0.35
    
    ax.bar(x - width/2, circular_accs, width, label='Circular', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, softmax_accs, width, label='Softmax', color='seagreen', alpha=0.8)
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Phase', fontsize=12)
    ax.set_title('Per-Phase Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(PHASE_NAMES)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_dir / "per_phase_accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confidence distribution (solo circular)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    confidence = circular_results['confidence']
    correct_mask = circular_results['predictions'] == circular_results['labels']
    
    ax.hist(confidence[correct_mask], bins=30, alpha=0.7, label='Correct', color='green')
    ax.hist(confidence[~correct_mask], bins=30, alpha=0.7, label='Incorrect', color='red')
    
    ax.axvline(confidence.mean(), color='black', linestyle='--', linewidth=2, 
               label=f'Mean: {confidence.mean():.3f}')
    ax.set_xlabel('Confidence (radius)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Circular Model: Confidence Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "circular_confidence_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Visualizaciones guardadas en: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Comparar Circular vs Softmax")
    
    parser.add_argument("--tensors-root", type=str, default="outputs_x3/tensors_15ch")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="outputs/model_comparison")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("🔵 CIRCULAR vs 📊 SOFTMAX - MODEL COMPARISON")
    print("="*60)
    print(f"Tensors root: {args.tensors_root}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # Load data
    print("\n📂 Loading data...")
    train_loader, val_loader = create_dataloaders(
        tensors_root=args.tensors_root,
        batch_size=args.batch_size,
        use_augmentation=True,
        num_workers=4,
    )
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    
    # Train both models
    circular_model, circular_history, circular_best = train_circular_model(
        train_loader, val_loader, args.epochs, args.device, args.lr
    )
    
    softmax_model, softmax_history, softmax_best = train_softmax_model(
        train_loader, val_loader, args.epochs, args.device, args.lr
    )
    
    # Evaluate both models
    print("\n📊 Evaluating models on validation set...")
    circular_results = evaluate_circular(circular_model, val_loader, args.device)
    softmax_results = evaluate_softmax(softmax_model, val_loader, args.device)
    
    # Compute metrics
    circular_acc = (circular_results['predictions'] == circular_results['labels']).mean()
    softmax_acc = (softmax_results['predictions'] == softmax_results['labels']).mean()
    
    print("\n" + "="*60)
    print("📈 FINAL RESULTS")
    print("="*60)
    print(f"🔵 Circular Model:")
    print(f"   Val Accuracy: {circular_acc:.3f} ({circular_acc*100:.1f}%)")
    print(f"   Mean Confidence: {circular_results['confidence'].mean():.3f}")
    print(f"   Std Confidence: {circular_results['confidence'].std():.3f}")
    
    print(f"\n📊 Softmax Model:")
    print(f"   Val Accuracy: {softmax_acc:.3f} ({softmax_acc*100:.1f}%)")
    
    # Winner
    print("\n" + "="*60)
    if circular_acc > softmax_acc:
        diff = (circular_acc - softmax_acc) * 100
        print(f"🏆 WINNER: Circular Model (+{diff:.1f}%)")
    elif softmax_acc > circular_acc:
        diff = (softmax_acc - circular_acc) * 100
        print(f"🏆 WINNER: Softmax Model (+{diff:.1f}%)")
    else:
        print(f"🤝 TIE!")
    print("="*60)
    
    # Generate visualizations
    print("\n📊 Generating comparison plots...")
    plot_comparison(circular_results, softmax_results, output_dir)
    
    # Save results
    comparison_results = {
        'timestamp': datetime.now().isoformat(),
        'epochs': args.epochs,
        'circular': {
            'val_accuracy': float(circular_acc),
            'mean_confidence': float(circular_results['confidence'].mean()),
            'std_confidence': float(circular_results['confidence'].std()),
        },
        'softmax': {
            'val_accuracy': float(softmax_acc),
        },
        'winner': 'circular' if circular_acc > softmax_acc else 'softmax' if softmax_acc > circular_acc else 'tie',
    }
    
    results_path = output_dir / "comparison_results.json"
    with open(results_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_path}")
    print(f"📊 Visualizations in: {output_dir}")


if __name__ == "__main__":
    main()
