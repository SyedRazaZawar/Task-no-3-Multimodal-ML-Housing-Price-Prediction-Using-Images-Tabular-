"""
Multimodal Machine Learning: Housing Price Prediction Using Images & Tabular Data

This script provides a complete, self-contained pipeline for predicting house prices
by combining structured tabular features with image features extracted via a CNN.
Since it is self-contained, running this script will:
1. Synthesize a mock housing dataset (tabular CSV + image directory).
2. Construct and train a PyTorch multimodal model (CNN + MLP fusion).
3. Evaluate the model using MAE, RMSE, and R^2.
4. Plot and save training history and regression performance.
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# Try importing ML dependencies with descriptive errors if missing
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
except ImportError:
    print("\n[!] PyTorch or torchvision is not installed. Please run:")
    print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    print("    (or run without '--index-url' for GPU support if available)\n")
    raise

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError:
    print("\n[!] scikit-learn is not installed. Please run:")
    print("    pip install scikit-learn\n")
    raise


# =====================================================================
# 1. SYNTHETIC DATA GENERATION (Tabular + Images)
# =====================================================================

def generate_synthetic_dataset(data_dir="dataset", num_samples=300):
    """
    Generates a synthetic dataset containing tabular characteristics of houses
    along with drawn 2D images representing their layouts. To simulate real
    multimodal relations, the house visual features (e.g., size, color, pool status)
    align directly with the tabular data, and both determine the price.
    """
    os.makedirs(data_dir, exist_ok=True)
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"[*] Generating {num_samples} synthetic house listings (CSV + images)...")
    
    records = []
    
    for i in range(num_samples):
        # Tabular attributes
        area_sqft = random.randint(800, 4500)
        num_bedrooms = random.randint(1, 5)
        num_bathrooms = random.randint(1, 4)
        age_years = random.randint(0, 50)
        has_pool = random.choice([0, 1])
        has_garage = random.choice([0, 1])
        # quality_grade: 1 (luxurious) to 5 (needs work)
        quality_grade = random.randint(1, 5)
        
        # We save visual characteristics in the image
        img_filename = f"house_{i:04d}.png"
        img_path = os.path.join(images_dir, img_filename)
        
        # Draw the house image
        img = Image.new("RGB", (128, 128), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        # 1. Draw Ground/Sky separator
        draw.rectangle([(0, 95), (128, 128)], fill=(120, 180, 120))  # Green lawn
        
        # 2. Draw House Body (size scales with area_sqft)
        house_width = int(40 + (area_sqft / 4500) * 50)
        house_height = int(35 + (area_sqft / 4500) * 35)
        left = 64 - (house_width // 2)
        right = 64 + (house_width // 2)
        top = 95 - house_height
        bottom = 95
        
        # House body color corresponds to overall condition/quality
        # Quality 1 = Premium brick color, Quality 5 = Dull grey
        colors = {
            1: (200, 80, 80),   # Red brick
            2: (220, 180, 120),  # Warm beige
            3: (220, 220, 180),  # Yellow cream
            4: (170, 170, 170),  # Grey
            5: (120, 120, 120)   # Dark dull grey
        }
        body_color = colors[quality_grade]
        draw.rectangle([(left, top), (right, bottom)], fill=body_color, outline=(40, 40, 40), width=2)
        
        # 3. Draw Roof (Style depends on quality grade)
        if quality_grade <= 2:
            # Luxury peaked triangle roof
            draw.polygon([(left - 5, top), (64, top - 25), (right + 5, top)], fill=(110, 50, 50), outline=(40, 40, 40))
        else:
            # Flat roof style
            draw.rectangle([(left - 2, top - 6), (right + 2, top)], fill=(80, 80, 80), outline=(40, 40, 40))
            
        # 4. Draw Windows (Quantity correlates with bedrooms)
        window_size = 8
        if num_bedrooms >= 1:
            # Window Left
            draw.rectangle([(left + 8, top + 10), (left + 8 + window_size, top + 10 + window_size)], fill=(200, 230, 255), outline=(40, 40, 40))
        if num_bedrooms >= 3:
            # Window Right
            draw.rectangle([(right - 8 - window_size, top + 10), (right - 8, top + 10 + window_size)], fill=(200, 230, 255), outline=(40, 40, 40))
        if num_bedrooms >= 4:
            # Top Window if peaked roof
            draw.rectangle([(64 - window_size//2, top - 12), (64 + window_size//2, top - 12 + window_size)], fill=(200, 230, 255), outline=(40, 40, 40))
            
        # 5. Draw Door
        door_w, door_h = 10, 18
        draw.rectangle([(64 - door_w//2, bottom - door_h), (64 + door_w//2, bottom)], fill=(90, 60, 40), outline=(40, 40, 40))
        
        # 6. Draw Pool if has_pool = 1 (Blue pond next to house)
        if has_pool == 1:
            draw.ellipse([(right + 5, 105), (right + 25, 120)], fill=(80, 180, 255), outline=(30, 100, 200))
            
        # 7. Draw Garage if has_garage = 1
        if has_garage == 1:
            garage_left = left - 15
            draw.rectangle([(garage_left, bottom - 22), (left, bottom)], fill=(150, 150, 150), outline=(40, 40, 40))
            # Garage doors lines
            draw.line([(garage_left + 2, bottom - 18), (left - 2, bottom - 18)], fill=(80, 80, 80))
            draw.line([(garage_left + 2, bottom - 10), (left - 2, bottom - 10)], fill=(80, 80, 80))
            
        # Save image
        img.save(img_path)
        
        # Base Price function with correlation + non-linearity + noise
        # Base price depends on area, quality, pool, garage, bedrooms, bathrooms, age
        base_price = 80000.0
        base_price += area_sqft * 160.0
        base_price += num_bedrooms * 18000.0
        base_price += num_bathrooms * 12000.0
        base_price += has_pool * 55000.0
        base_price += has_garage * 30000.0
        base_price -= age_years * 1200.0
        
        # Quality score premium (lower grade index = higher quality)
        quality_premium = {1: 90000, 2: 50000, 3: 20000, 4: 0, 5: -15000}
        base_price += quality_premium[quality_grade]
        
        # Visual premium (simulate extraction of visual clues like large lawns or nice roofs)
        # Adding visual-only correlation (e.g. brick color premium)
        if quality_grade == 1:
            base_price += 15000.0 # Red brick aesthetics
            
        # Add random noise (representing other market factors)
        noise = random.normalvariate(0, 12000)
        final_price = max(40000.0, base_price + noise)
        
        records.append({
            "image_path": img_filename,
            "area_sqft": area_sqft,
            "num_bedrooms": num_bedrooms,
            "num_bathrooms": num_bathrooms,
            "age_years": age_years,
            "has_pool": has_pool,
            "has_garage": has_garage,
            "quality_grade": quality_grade,
            "price": round(final_price, 2)
        })
        
    df = pd.DataFrame(records)
    csv_path = os.path.join(data_dir, "housing_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"[+] Dataset saved. CSV path: {csv_path}. Images directory: {images_dir}\n")
    return csv_path, images_dir


# =====================================================================
# 2. PYTORCH MULTIMODAL DATASET
# =====================================================================

class MultimodalHousingDataset(Dataset):
    """
    Custom PyTorch Dataset class that handles loading tabular data and
    associated images, applying respective transformations.
    """
    def __init__(self, csv_file, img_dir, tabular_scaler=None, transform=None, is_train=True):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Select tabular feature columns
        self.tabular_cols = [
            "area_sqft", "num_bedrooms", "num_bathrooms", 
            "age_years", "has_pool", "has_garage", "quality_grade"
        ]
        
        # Fit scaler on tabular columns if training, otherwise use passed scaler
        if is_train:
            self.scaler = StandardScaler()
            self.tabular_features = self.scaler.fit_transform(self.df[self.tabular_cols])
        else:
            self.scaler = tabular_scaler
            self.tabular_features = self.scaler.transform(self.df[self.tabular_cols])
            
        self.prices = self.df["price"].values.astype(np.float32)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        # 1. Load image
        img_name = self.df.iloc[idx]["image_path"]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # 2. Get tabular features
        tabular = torch.tensor(self.tabular_features[idx], dtype=torch.float32)
        
        # 3. Get target price
        price = torch.tensor(self.prices[idx], dtype=torch.float32)
        
        return {
            "image": image,
            "tabular": tabular
        }, price


# =====================================================================
# 3. MULTIMODAL NETWORK ARCHITECTURE
# =====================================================================

class CNNBranch(nn.Module):
    """
    CNN Branch to extract compact feature representations from housing images.
    """
    def __init__(self, output_dim=64):
        super(CNNBranch, self).__init__()
        # Input size: 3 x 128 x 128
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # -> 16 x 64 x 64
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # -> 32 x 32 x 32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # -> 64 x 16 x 16
        )
        
        self.pool = nn.AdaptiveAvgPool2d((4, 4)) # -> 64 x 4 x 4 = 1024 features
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class TabularBranch(nn.Module):
    """
    Multi-layer Perceptron (MLP) branch to process tabular numerical/categorical data.
    """
    def __init__(self, input_dim, output_dim=64):
        super(TabularBranch, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        return self.fc(x)


class MultimodalFusionModel(nn.Module):
    """
    Late Fusion architecture combining extracted CNN features with MLP tabular features
    to make final regression prediction of house price.
    """
    def __init__(self, num_tabular_features, image_feature_dim=64, tabular_feature_dim=64):
        super(MultimodalFusionModel, self).__init__()
        self.cnn_branch = CNNBranch(output_dim=image_feature_dim)
        self.tab_branch = TabularBranch(input_dim=num_tabular_features, output_dim=tabular_feature_dim)
        
        # Combined feature size = image features + tabular features
        combined_dim = image_feature_dim + tabular_feature_dim
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
        
    def forward(self, img_input, tab_input):
        # Forward pass on both branches
        img_feats = self.cnn_branch(img_input)
        tab_feats = self.tab_branch(tab_input)
        
        # Feature fusion (late concatenation)
        fused = torch.cat((img_feats, tab_feats), dim=1)
        
        # Predict price
        price = self.regressor(fused)
        return price.squeeze(-1) # return shape (batch_size,)


# =====================================================================
# 4. TRAINING & EVALUATION FUNCTIONS
# =====================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        img_in = inputs["image"].to(device)
        tab_in = inputs["tabular"].to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(img_in, tab_in)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * targets.size(0)
    return running_loss / len(dataloader.dataset)


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            img_in = inputs["image"].to(device)
            tab_in = inputs["tabular"].to(device)
            
            predictions = model(img_in, tab_in)
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate performance metrics
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    
    return mae, rmse, r2, all_preds, all_targets


# =====================================================================
# 5. MAIN PIPELINE EXECUTION
# =====================================================================

def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 1. Synthesize Data
    data_dir = "dataset"
    csv_file, img_dir = generate_synthetic_dataset(data_dir=data_dir, num_samples=500)
    
    # 2. Data Splits
    df = pd.read_csv(csv_file)
    train_idx, test_idx = train_test_split(df.index, test_size=0.20, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42) # 15% of 80% is 12% val
    
    # Write split CSVs to separate training loaders
    train_csv = os.path.join(data_dir, "train_data.csv")
    val_csv = os.path.join(data_dir, "val_data.csv")
    test_csv = os.path.join(data_dir, "test_data.csv")
    
    df.loc[train_idx].to_csv(train_csv, index=False)
    df.loc[val_idx].to_csv(val_csv, index=False)
    df.loc[test_idx].to_csv(test_csv, index=False)
    
    # 3. Define Image Transforms
    # Resizing, normalizing, and simple train augmentation (flips)
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 4. Initialize Custom Datasets and Dataloaders
    # Fitting the standard scaler on training, applying it to validation/test
    train_dataset = MultimodalHousingDataset(train_csv, img_dir, transform=train_transform, is_train=True)
    scaler = train_dataset.scaler # Keep scaler for inference preprocessing
    
    val_dataset = MultimodalHousingDataset(val_csv, img_dir, tabular_scaler=scaler, transform=val_test_transform, is_train=False)
    test_dataset = MultimodalHousingDataset(test_csv, img_dir, tabular_scaler=scaler, transform=val_test_transform, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Get feature size
    num_tabular_feats = len(train_dataset.tabular_cols)
    
    # 5. Build Model and Set Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device}")
    
    model = MultimodalFusionModel(num_tabular_features=num_tabular_feats).to(device)
    
    # Define Loss function & Optimizer
    # We use MSE Loss for regression task
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # Cosine scheduler to decay learning rate over training epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    # 6. Training Loop
    epochs = 40
    train_losses = []
    val_losses = []
    
    print(f"[*] Training Multimodal Fusion Network for {epochs} epochs...")
    best_val_mae = float("inf")
    best_model_path = "best_multimodal_model.pth"
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        # Validate (expressed in Root MSE loss value)
        val_mae, val_rmse, val_r2, _, _ = evaluate_model(model, val_loader, device)
        val_loss = val_rmse ** 2 # MSE
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), best_model_path)
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:02d}/{epochs:02d} | Train MSE: {train_loss:.2e} | Val RMSE: {val_rmse:,.2f} | Val MAE: ${val_mae:,.2f} | Val R^2: {val_r2:.4f}")
            
    print(f"[+] Training complete. Best model weights saved to: '{best_model_path}'")
    
    # 7. Final Model Evaluation on Unseen Test Dataset
    # Load best model weights
    model.load_state_dict(torch.load(best_model_path))
    test_mae, test_rmse, test_r2, predictions, targets = evaluate_model(model, test_loader, device)
    
    print("\n" + "="*50)
    print("                EVALUATION METRICS                ")
    print("="*50)
    print(f"Mean Absolute Error (MAE):     ${test_mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${test_rmse:,.2f}")
    print(f"R-squared Coefficient (R^2):    {test_r2:.4f}")
    print("="*50 + "\n")
    
    # 8. Plot and Save Visualizations
    plot_results(train_losses, val_losses, targets, predictions)


def plot_results(train_losses, val_losses, targets, predictions):
    """
    Helper function to generate evaluation curves and scatter plots,
    saving them as local image artifacts.
    """
    plt.figure(figsize=(14, 5))
    
    # Subplot 1: Train & Validation Losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss (MSE)", color="dodgerblue", lw=2)
    plt.plot(val_losses, label="Validation Loss (MSE)", color="coral", lw=2)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE log-scaled)")
    plt.title("Training and Validation Performance")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Subplot 2: Predictions vs. Ground Truth
    plt.subplot(1, 2, 2)
    max_val = max(max(targets), max(predictions))
    min_val = min(min(targets), min(predictions))
    
    plt.scatter(targets, predictions, alpha=0.6, color="purple", edgecolors="w", s=40, label="Predictions")
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Alignment")
    plt.xlabel("Actual Price ($)")
    plt.ylabel("Predicted Price ($)")
    plt.title("Predictions vs. Ground Truth Prices")
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plots_path = "multimodal_evaluation_plots.png"
    plt.savefig(plots_path, dpi=150)
    plt.close()
    
    print(f"[+] Evaluation plots saved to: {os.path.abspath(plots_path)}")


if __name__ == "__main__":
    main()
