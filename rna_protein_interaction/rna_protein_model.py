import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# 1. DATA PREPARATION
# =============================================================================

class RNAProteinDataset(Dataset):
    """Dataset for RNA-Protein interaction prediction"""
    
    def __init__(self, rna_sequences, protein_sequences, labels):
        self.rna_sequences = rna_sequences
        self.protein_sequences = protein_sequences
        self.labels = labels
        
        # Encoding dictionaries
        self.rna_vocab = {'A': 1, 'U': 2, 'G': 3, 'C': 4, 'N': 0}  # N for padding
        self.protein_vocab = {
            'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
            'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
            'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, 'X': 0  # X for padding
        }
    
    def encode_sequence(self, sequence, vocab, max_len):
        """Encode a sequence into numerical representation"""
        encoded = [vocab.get(char, 0) for char in sequence[:max_len]]
        # Pad if necessary
        if len(encoded) < max_len:
            encoded += [0] * (max_len - len(encoded))
        return torch.tensor(encoded, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        rna = self.encode_sequence(self.rna_sequences[idx], self.rna_vocab, max_len=100)
        protein = self.encode_sequence(self.protein_sequences[idx], self.protein_vocab, max_len=200)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return rna, protein, label


def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic RNA and protein sequences for demonstration
    In practice, you'd load real data from a database like RPI
    """
    rna_bases = ['A', 'U', 'G', 'C']
    amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
    
    rna_sequences = []
    protein_sequences = []
    labels = []
    
    for i in range(n_samples):
        # Generate random sequences
        rna_len = np.random.randint(30, 100)
        protein_len = np.random.randint(50, 200)
        
        rna = ''.join(np.random.choice(rna_bases, size=rna_len))
        protein = ''.join(np.random.choice(amino_acids, size=protein_len))
        
        # Create synthetic labels (in reality, these come from experimental data)
        # Simple rule: if both sequences have certain patterns, they "interact"
        label = 1 if (rna.count('GGG') > 0 and protein.count('RRR') > 0) else 0
        # Add some randomness to make it realistic
        if np.random.random() < 0.3:
            label = 1 - label
        
        rna_sequences.append(rna)
        protein_sequences.append(protein)
        labels.append(label)
    
    return rna_sequences, protein_sequences, labels


# =============================================================================
# 2. MODEL ARCHITECTURE
# =============================================================================

class RNAProteinInteractionModel(nn.Module):
    """
    Neural network for predicting RNA-Protein interactions
    Uses separate embedding and CNN layers for RNA and protein sequences
    """
    
    def __init__(self, rna_vocab_size=5, protein_vocab_size=21, 
                 embedding_dim=64, hidden_dim=128):
        super(RNAProteinInteractionModel, self).__init__()
        
        # Embeddings for RNA and protein sequences
        self.rna_embedding = nn.Embedding(rna_vocab_size, embedding_dim, padding_idx=0)
        self.protein_embedding = nn.Embedding(protein_vocab_size, embedding_dim, padding_idx=0)
        
        # Convolutional layers for feature extraction
        self.rna_conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1)
        self.rna_conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        
        self.protein_conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=5, padding=2)
        self.protein_conv2 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        
        # Pooling
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 + 64, hidden_dim)  # Combined features
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, rna, protein):
        # Embed sequences
        rna_emb = self.rna_embedding(rna)  # (batch, seq_len, embed_dim)
        protein_emb = self.protein_embedding(protein)
        
        # Transpose for Conv1d (batch, embed_dim, seq_len)
        rna_emb = rna_emb.transpose(1, 2)
        protein_emb = protein_emb.transpose(1, 2)
        
        # Apply convolutions
        rna_feat = self.relu(self.rna_conv1(rna_emb))
        rna_feat = self.relu(self.rna_conv2(rna_feat))
        rna_feat = self.pool(rna_feat).squeeze(-1)  # (batch, 64)
        
        protein_feat = self.relu(self.protein_conv1(protein_emb))
        protein_feat = self.relu(self.protein_conv2(protein_feat))
        protein_feat = self.pool(protein_feat).squeeze(-1)  # (batch, 64)
        
        # Combine features
        combined = torch.cat([rna_feat, protein_feat], dim=1)
        
        # Fully connected layers
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        
        return x.squeeze()


# =============================================================================
# 3. TRAINING FUNCTION
# =============================================================================

def train_model(model, train_loader, val_loader, epochs=20, lr=0.001):
    """Train the RNA-Protein interaction model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for rna, protein, labels in train_loader:
            rna, protein, labels = rna.to(device), protein.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(rna, protein)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for rna, protein, labels in val_loader:
                rna, protein, labels = rna.to(device), protein.to(device), labels.to(device)
                
                outputs = model(rna, protein)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        accuracy = accuracy_score(all_labels, all_preds)
        val_accuracies.append(accuracy)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}")
    
    return train_losses, val_losses, val_accuracies


# =============================================================================
# 4. EVALUATION FUNCTION
# =============================================================================

def evaluate_model(model, test_loader):
    """Evaluate the model on test data"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for rna, protein, labels in test_loader:
            rna, protein, labels = rna.to(device), protein.to(device), labels.to(device)
            
            outputs = model(rna, protein)
            preds = (outputs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    print("\n=== Test Set Results ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    return accuracy, precision, recall, f1


# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

def main():
    print("RNA-Protein Interaction Predictor")
    print("=" * 50)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    rna_seqs, protein_seqs, labels = generate_synthetic_data(n_samples=1000)
    print(f"Generated {len(labels)} samples")
    print(f"Positive samples: {sum(labels)}, Negative samples: {len(labels) - sum(labels)}")
    
    # Split data
    print("\n2. Splitting data into train/val/test sets...")
    rna_train, rna_temp, protein_train, protein_temp, y_train, y_temp = train_test_split(
        rna_seqs, protein_seqs, labels, test_size=0.3, random_state=42
    )
    rna_val, rna_test, protein_val, protein_test, y_val, y_test = train_test_split(
        rna_temp, protein_temp, y_temp, test_size=0.5, random_state=42
    )
    
    # Create datasets and dataloaders
    train_dataset = RNAProteinDataset(rna_train, protein_train, y_train)
    val_dataset = RNAProteinDataset(rna_val, protein_val, y_val)
    test_dataset = RNAProteinDataset(rna_test, protein_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    print("\n3. Initializing model...")
    model = RNAProteinInteractionModel()
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    print("\n4. Training model...")
    train_losses, val_losses, val_accs = train_model(
        model, train_loader, val_loader, epochs=20, lr=0.001
    )
    
    # Evaluate on test set
    print("\n5. Evaluating on test set...")
    evaluate_model(model, test_loader)
    
    # Plot training curves
    print("\n6. Generating training curves...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("Training curves saved as 'training_curves.png'")
    
    print("\n" + "=" * 50)
    print("✓ Model training and evaluation complete!")
    print("\nThis demonstrates:")
    print("  - Loading and preprocessing sequence data")
    print("  - Building a deep learning model with embeddings and CNNs")
    print("  - Training with proper train/val/test splits")
    print("  - Evaluating with standard metrics")

if __name__ == "__main__":
    main()