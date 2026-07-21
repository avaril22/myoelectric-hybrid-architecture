"""
Training utilities including EWC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ==================== Standard Training ====================

def train_epoch(model, train_loader, optimizer, device='cuda'):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(train_loader), 100. * correct / total


def evaluate(model, data_loader, device='cuda'):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(data_loader), 100. * correct / total


# ==================== EWC ====================

def compute_fisher(model, data_loader, device='cuda', num_samples=None):
    """Compute Fisher Information Matrix diagonal."""
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    
    num_batches = len(data_loader) if num_samples is None else num_samples // data_loader.batch_size
    
    for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader, desc="Computing Fisher", total=num_batches)):
        if num_samples and batch_idx >= num_batches:
            break
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        model.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.data.pow(2)
    
    for n in fisher:
        fisher[n] /= num_batches
    
    return fisher


def ewc_loss(model, fisher, old_params, lambda_ewc=1000.0):
    """Compute EWC regularization loss."""
    loss = 0.0
    for n, p in model.named_parameters():
        if n in fisher and p.requires_grad:
            loss += (fisher[n] * (p - old_params[n]).pow(2)).sum()
    return lambda_ewc * loss / 2.0


def train_with_ewc(model, train_loader, val_loader, fisher=None, old_params=None,
                   lambda_ewc=1000.0, lr=0.001, epochs=50, device='cuda', patience=10):
    """Train with EWC regularization."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    use_ewc = (fisher is not None) and (old_params is not None)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            ce_loss = F.cross_entropy(outputs, targets)
            
            if use_ewc:
                loss = ce_loss + ewc_loss(model, fisher, old_params, lambda_ewc)
            else:
                loss = ce_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += ce_loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}: Train {train_acc:.2f}% | Val {val_acc:.2f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return history, best_val_acc