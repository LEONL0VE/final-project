
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import iTransformer model
# Ensure the directory is in path if needed, but since we are in the root of the project it should be fine
from models import iTransformer
from utils.timefeatures import time_features

### 1. Data Loading and Preprocessing ###

def load_and_process_data(mode='B', root_path='汇总/汇总'):
    """
    mode 'B': Train on V1-V7, Test on V8-V10
    """
    
    train_files = []
    test_files = []
    
    if mode == 'B':
        # V1 to V7 for training
        for i in range(1, 8):
            train_files.append(f'concatenated_result917_V{i}.csv')
        # V8 to V10 for testing
        for i in range(8, 11):
            test_files.append(f'concatenated_result917_V{i}.csv')
            
    # Helper to load and concat
    def load_files(file_list):
        dfs = []
        for f in file_list:
            path = os.path.join(root_path, f)
            if os.path.exists(path):
                df = pd.read_csv(path)
                dfs.append(df)
            else:
                print(f"Warning: File {path} not found.")
        if not dfs:
            return None
        return pd.concat(dfs, ignore_index=True)

    df_train = load_files(train_files)
    df_test = load_files(test_files)
    
    return df_train, df_test

def create_dataset_loader(df_train, df_test, window_size=30, pred_len=1, batch_size=32):
    # Prepare features and target
    # Target is the last column
    # We also need date for time_features
    
    # Extract features (exclude date)
    train_data_raw = df_train.drop(columns=['date']).values
    test_data_raw = df_test.drop(columns=['date']).values
    
    scaler = MinMaxScaler()
    # Fit ONLY on training data
    train_data_norm = scaler.fit_transform(train_data_raw)
    test_data_norm = scaler.transform(test_data_raw)
    
    # Process Time Features
    def get_time_mark(df):
        df_stamp = df[['date']].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        data_stamp = time_features(df_stamp, timeenc=1, freq='B') 
        return data_stamp
    
    train_mark = get_time_mark(df_train)
    test_mark = get_time_mark(df_test)
    
    # Sliding Window Generator
    def create_sliding_window(data, mark, window, length_size):
        seq_len = window
        sequence_length = seq_len + length_size
        result = []
        result_mark = []
        for i in range(len(data) - sequence_length + 1):
            result.append(data[i: i + sequence_length])
            result_mark.append(mark[i: i + sequence_length])
            
        result = np.array(result)
        result_mark = np.array(result_mark)
        
        x = result[:, :-length_size]
        y = result[:, -(length_size + int(window/2)):] # This output shape logic from original code seems specific to iTransformer
        # Note: Standard LSTM/CNN usually predicts just the target at the next step(s).
        # iTransformer often predicts a sequence. 
        # For fair comparison, we will adjust the label shape for standard models in the training loop.
        
        x_mark = result_mark[:, :-length_size]
        y_mark = result_mark[:, -(length_size + int(window/2)):]
        
        return (torch.tensor(x).float(), torch.tensor(y).float(), 
                torch.tensor(x_mark).float(), torch.tensor(y_mark).float())

    x_train, y_train, x_train_mark, y_train_mark = create_sliding_window(train_data_norm, train_mark, window_size, pred_len)
    x_test, y_test, x_test_mark, y_test_mark = create_sliding_window(test_data_norm, test_mark, window_size, pred_len)
    
    train_loader = DataLoader(TensorDataset(x_train, y_train, x_train_mark, y_train_mark), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test, x_test_mark, y_test_mark), batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler

### 2. Model Definitions ###

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()


# 2. Optimized CNN-LSTM (Enhanced Capacity + Bi-Directional Option)
class CNNLSTM(BaseModel):
    def __init__(self, input_dim, seq_len, hidden_dim=128, output_len=1):
        super().__init__()
        # Keep Kernel size 3, but increase filters
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1) # Reduced dropout
        
        # Use Bidirectional LSTM to capture more context like the successful Bi-GRU
        self.lstm = nn.LSTM(64, hidden_dim, num_layers=2, batch_first=True, 
                           dropout=0.1, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, output_len) # *2 for bidirectional

    def forward(self, x, x_mark, y, y_mark, mask=None):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = x.permute(0, 2, 1) # Back to (Batch, Seq, Dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Last hidden state of the sequence
        out = self.linear(out)
        return out.unsqueeze(-1)

# 3. Bidirectional GRU (Optimized GRU)
class BiGRUModel(BaseModel):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_len=1):
        super().__init__()
        # Bidirectional=True doubles the hidden info
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, 
                          dropout=0.1, bidirectional=True) # Reduced dropout
        self.linear = nn.Linear(hidden_dim * 2, output_len)

    def forward(self, x, x_mark, y, y_mark, mask=None):
        out, _ = self.gru(x)
        out = out[:, -1, :] 
        out = self.linear(out)
        return out.unsqueeze(-1)

# 4. CNN-BiGRU (Upgraded from CNN-GRU)
class CNNGRU(BaseModel):
    def __init__(self, input_dim, seq_len, hidden_dim=128, output_len=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1) # Reduced dropout
        
        # Upgraded to Bidirectional GRU
        self.gru = nn.GRU(64, hidden_dim, num_layers=2, batch_first=True, 
                          dropout=0.1, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, output_len)

    def forward(self, x, x_mark, y, y_mark, mask=None):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.linear(out)
        return out.unsqueeze(-1)


### 3. Training Loop ###

def train_model(model, train_loader, criterion, optimizer, device, epochs, pred_len):
    model.train()
    loss_history = []
    
    for epoch in range(epochs):
        total_loss = 0
        for x, y, x_mark, y_mark in train_loader:
            x, y, x_mark, y_mark = x.to(device), y.to(device), x_mark.to(device), y_mark.to(device)
            
            optimizer.zero_grad()
            
            # iTransformer and custom models have slightly different signatures in provided code
            # But I unified my custom models to accept (x, x_mark, y, y_mark)
            # iTransformer's forward also accepts these.
            
            preds = model(x, x_mark, y, y_mark)
            
            # Target adjustments
            # y might contain label_len context + pred_len
            # We only care about the last 'pred_len' for loss
            
            if hasattr(model, 'task_name') and model.task_name == 'short_term_forecast':
                 # iTransformer output shape: [Batch, Pred_Len, N_Vars]
                 # If N_Vars > 1, we need to select the target variable (assume it's the last one)
                 if preds.dim() == 3 and preds.shape[2] > 1:
                     preds = preds[:, :, -1]
                 
                 preds = preds.squeeze()
                 target = y[:, -pred_len:, -1].squeeze() # Assuming target is last column
            else:
                 # Standard models return (Batch, Output_Len, 1) or (Batch, Output_Len)
                 if preds.dim() == 3 and preds.shape[2] == 1:
                     preds = preds.squeeze(-1)
                     
                 preds = preds.squeeze()
                 target = y[:, -pred_len:, -1].squeeze()
            
            # Ensure shapes match exactly before loss
            if preds.shape != target.shape:
                # Handle edge case where batch size is 1 or pred_len is 1 and squeeze removed too much
                # Reshape to [Batch, Pred_Len] if possible, or just flatten both
                preds = preds.reshape(-1)
                target = target.reshape(-1)

            loss = criterion(preds, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
        loss_history.append(total_loss/len(train_loader))
        
    return loss_history

def evaluate_model(model, test_loader, scaler, device, pred_len):
    model.eval()
    preds_list = []
    trues_list = []
    
    with torch.no_grad():
        for x, y, x_mark, y_mark in test_loader:
            x, y, x_mark, y_mark = x.to(device), y.to(device), x_mark.to(device), y_mark.to(device)
            
            preds = model(x, x_mark, y, y_mark)
            
            # Handling multivariate output for iTransformer
            if preds.dim() == 3 and preds.shape[2] > 1:
                preds = preds[:, :, -1]
            
            # Assuming single step prediction for simplicity in this specific experiment request
            preds = preds.squeeze().cpu().numpy()
            target = y[:, -pred_len:, -1].squeeze().cpu().numpy()
            
            # Handle scalar/0-d array issues if batch size is 1
            if np.ndim(preds) == 0:
                preds = np.expand_dims(preds, axis=0)
            if np.ndim(target) == 0:
                target = np.expand_dims(target, axis=0)
                
            preds_list.extend(preds)
            trues_list.extend(target)
            
    # Inverse transform
    # The scaler was fit on all columns. We need to handle this.
    # The target was the LAST column of the data used to fit scaler.
    # We need to construct a dummy matrix to inverse transform.
    
    # This is a bit tricky. The cleanest way if we only care about target is:
    # Scale -> Train -> Predict Scaled -> Inverse Scale just the target column?
    # BUT sklearn MinMaxScaler inverse_transform requires the same shape (n_features).
    
    # Workaround: Manually use min/max of the target column.
    
    target_col_idx = -1
    data_min = scaler.data_min_[target_col_idx]
    data_max = scaler.data_max_[target_col_idx]
    
    def manual_inverse(data_norm, min_val, max_val):
        return data_norm * (max_val - min_val) + min_val
    
    preds_inv = manual_inverse(np.array(preds_list), data_min, data_max)
    trues_inv = manual_inverse(np.array(trues_list), data_min, data_max)
    
    # Metric
    mse = mean_squared_error(trues_inv, preds_inv)
    mae = mean_absolute_error(trues_inv, preds_inv)
    r2 = r2_score(trues_inv, preds_inv)
    
    return {
        "MSE": mse,
        "MAE": mae,
        "R2": r2,
        "Preds": preds_inv,
        "Trues": trues_inv
    }

### 4. Main Execution ###

def main():
    root_path = '汇总/汇总'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Loading Data (V1-V7 Train, V8-V10 Test)...")
    df_train, df_test = load_and_process_data('B', root_path)
    if df_train is None or df_test is None:
        print("Error: Data files not found.")
        return

    # Configuration - Updated to match LSTM+iTransformer best settings
    WINDOW_SIZE = 9
    PRED_LEN = 1
    BATCH_SIZE = 64
    INPUT_DIM = df_train.shape[1] - 1 # Subtract date
    EPOCHS = 50 # Increased from 20 to 50 for better convergence
    LR = 0.001
    
    # Create results directory
    result_dir = 'experiment_results'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    print(f"Results will be saved to: {result_dir}")
    
    # Create Loaders
    train_loader, test_loader, scaler = create_dataset_loader(df_train, df_test, WINDOW_SIZE, PRED_LEN, BATCH_SIZE)
    
    results = {}
    
    # ------------------
    # Model 1: Optimized CNN-BiLSTM (Improved)
    # ------------------
    print("\nRunning Optimized CNN-BiLSTM...")
    model_cnnlstm = CNNLSTM(input_dim=INPUT_DIM, seq_len=WINDOW_SIZE, output_len=PRED_LEN).to(device)
    optimizer = optim.Adam(model_cnnlstm.parameters(), lr=LR)
    criterion = nn.MSELoss()
    train_model(model_cnnlstm, train_loader, criterion, optimizer, device, EPOCHS, PRED_LEN)
    res_cnnlstm = evaluate_model(model_cnnlstm, test_loader, scaler, device, PRED_LEN)
    results['CNN-BiLSTM'] = res_cnnlstm
    print(f"CNN-BiLSTM Results: {res_cnnlstm['MSE']}, R2: {res_cnnlstm['R2']}")
    
    # ------------------
    # Model 2: Bidirectional GRU
    # ------------------
    print("\nRunning Bi-GRU...")
    model_gru = BiGRUModel(input_dim=INPUT_DIM, output_len=PRED_LEN).to(device)
    optimizer = optim.Adam(model_gru.parameters(), lr=LR)
    criterion = nn.MSELoss()
    train_model(model_gru, train_loader, criterion, optimizer, device, EPOCHS, PRED_LEN)
    res_gru = evaluate_model(model_gru, test_loader, scaler, device, PRED_LEN)
    results['Bi-GRU'] = res_gru
    print(f"Bi-GRU Results: {res_gru['MSE']}, R2: {res_gru['R2']}")

    # ------------------
    # Model 3: CNN-BiGRU (Improved)
    # ------------------
    print("\nRunning CNN-BiGRU...")
    model_cnngru = CNNGRU(input_dim=INPUT_DIM, seq_len=WINDOW_SIZE, output_len=PRED_LEN).to(device)
    optimizer = optim.Adam(model_cnngru.parameters(), lr=LR)
    criterion = nn.MSELoss()
    train_model(model_cnngru, train_loader, criterion, optimizer, device, EPOCHS, PRED_LEN)
    res_cnngru = evaluate_model(model_cnngru, test_loader, scaler, device, PRED_LEN)
    results['CNN-BiGRU'] = res_cnngru

    print(f"CNN-GRU Results: {res_cnngru['MSE']}, R2: {res_cnngru['R2']}")
    
    # ------------------
    # Model 4: LSTM + Improved iTransformer (Best Model Config)
    # ------------------
    print("\nRunning LSTM+Improved iTransformer...")
    
    # Config matching LSTM+改进的itransformer.py
    class Config:
        def __init__(self):
            # basic
            self.seq_len = WINDOW_SIZE
            self.label_len = int(WINDOW_SIZE / 2)
            self.pred_len = PRED_LEN
            self.freq = 'b'
            
            # training
            self.batch_size = BATCH_SIZE
            self.num_epochs = EPOCHS
            self.learning_rate = LR
            
            # model define
            self.enc_in = INPUT_DIM
            self.dec_in = INPUT_DIM
            self.c_out = 1 
            self.d_model = 64
            self.n_heads = 8
            self.e_layers = 2
            self.d_layers = 1
            self.d_ff = 64
            self.factor = 5
            self.activation = 'gelu'
            self.channel_independence = 0
            
            # specific params
            self.top_k = 5
            self.num_kernels = 6
            self.distil = 1
            self.embedding = 'timeF' # Default usually
            self.embed = 'timeF'
            self.output_attention = 0
            self.task_name = 'short_term_forecast'
            self.moving_avg = WINDOW_SIZE - 1
            self.dropout = 0.1

    config = Config()
    model_itrans = iTransformer.Model(config).to(device)
    
    optimizer = optim.Adam(model_itrans.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    train_model(model_itrans, train_loader, criterion, optimizer, device, EPOCHS, PRED_LEN)
    res_itrans = evaluate_model(model_itrans, test_loader, scaler, device, PRED_LEN)
    results['LSTM_iTransformer_Improved'] = res_itrans
    print(f"LSTM+iTransformer Results: {res_itrans['MSE']}, R2: {res_itrans['R2']}")
    
    # Save Summary and Plot
    summary = []
    
    # Data for plotting
    plot_data = pd.DataFrame()
    # Add True values only once (they are same for all models)
    # Use the first model's true values
    first_model = list(results.keys())[0]
    plot_data['True'] = results[first_model]['Trues']
    
    for name, res in results.items():
        summary.append({
            "Model": name,
            "MSE": res['MSE'],
            "MAE": res['MAE'],
            "R2": res['R2']
        })
        
        # Add preds to plot data
        plot_data[name] = res['Preds']
        
        # Save individual prediction csvs
        df_pred = pd.DataFrame({
            "True": res['Trues'],
            "Pred": res['Preds']
        })
        df_pred.to_csv(os.path.join(result_dir, f"result_{name}_MultiCar.csv"), index=False)
        
    df_summary = pd.DataFrame(summary)
    summary_path = os.path.join(result_dir, "MultiCar_Experiment_Summary.csv")

    df_summary.to_csv(summary_path, index=False)
    
    # Plotting
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 6))
    # Plot a subset if data is too large, e.g., last 200 points
    plot_len = min(200, len(plot_data))
    subset = plot_data.iloc[-plot_len:]
    
    plt.plot(subset.index, subset['True'], label='True Value', color='black', linewidth=2)
    
    colors = ['blue', 'green', 'orange', 'red']
    for i, name in enumerate(results.keys()):
        color = colors[i % len(colors)]
        plt.plot(subset.index, subset[name], label=name, color=color, alpha=0.7)
        
    plt.title(f'Model Comparison (Last {plot_len} Time Steps)')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(result_dir, "Model_Comparison_Plot.png")
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")
    
    print("\nExperiment Complete. Results saved.")
    print(df_summary)

if __name__ == "__main__":
    main()
