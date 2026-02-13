
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
from models import iTransformer
from utils.timefeatures import time_features

### 1. Data Loading and Preprocessing (Single Car) ###

def load_data_single_car(car_id='V1', split_ratio=0.8, root_path='汇总/汇总'):
    """
    mode 'Single': Train on first 80% of V{i}, Test on last 20%
    """
    filename = f'concatenated_result917_{car_id}.csv'
    path = os.path.join(root_path, filename)
    
    if not os.path.exists(path):
        print(f"Error: File {path} not found.")
        return None, None

    df = pd.read_csv(path)
    
    # Split chronologically
    total_len = len(df)
    train_size = int(total_len * split_ratio)
    
    df_train = df.iloc[:train_size].copy()
    df_test = df.iloc[train_size:].copy()
    
    print(f"Data Loaded for {car_id}: Total {total_len}, Train {len(df_train)}, Test {len(df_test)}")
    return df_train, df_test

def create_dataset_loader(df_train, df_test, window_size=30, pred_len=1, batch_size=32):
    # Prepare features and target
    # Target is the last column (Target)
    
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
        
        # x: input sequence, y: label sequence (including pred_len)
        x = result[:, :-length_size]
        y = result[:, -(length_size + int(window/2)):] 
        
        x_mark = result_mark[:, :-length_size]
        y_mark = result_mark[:, -(length_size + int(window/2)):]
        
        return (torch.tensor(x).float(), torch.tensor(y).float(), 
                torch.tensor(x_mark).float(), torch.tensor(y_mark).float())

    x_train, y_train, x_train_mark, y_train_mark = create_sliding_window(train_data_norm, train_mark, window_size, pred_len)
    x_test, y_test, x_test_mark, y_test_mark = create_sliding_window(test_data_norm, test_mark, window_size, pred_len)
    
    train_loader = DataLoader(TensorDataset(x_train, y_train, x_train_mark, y_train_mark), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test, x_test_mark, y_test_mark), batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler

### 2. Model Definitions (Same Improved Models) ###

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

# 1. Optimized CNN-BiLSTM
class CNNLSTM(BaseModel):
    def __init__(self, input_dim, seq_len, hidden_dim=128, output_len=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        self.lstm = nn.LSTM(64, hidden_dim, num_layers=2, batch_first=True, 
                           dropout=0.1, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, output_len)

    def forward(self, x, x_mark, y, y_mark, mask=None):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.linear(out)
        return out.unsqueeze(-1)

# 2. Bidirectional GRU
class BiGRUModel(BaseModel):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_len=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, 
                          dropout=0.1, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, output_len)

    def forward(self, x, x_mark, y, y_mark, mask=None):
        out, _ = self.gru(x)
        out = out[:, -1, :] 
        out = self.linear(out)
        return out.unsqueeze(-1)

# 3. CNN-BiGRU
class CNNGRU(BaseModel):
    def __init__(self, input_dim, seq_len, hidden_dim=128, output_len=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
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
            preds = model(x, x_mark, y, y_mark)
            
            if hasattr(model, 'task_name') and model.task_name == 'short_term_forecast':
                 if preds.dim() == 3 and preds.shape[2] > 1:
                     preds = preds[:, :, -1]
                 preds = preds.squeeze()
                 target = y[:, -pred_len:, -1].squeeze()
            else:
                 if preds.dim() == 3 and preds.shape[2] == 1:
                     preds = preds.squeeze(-1)
                 preds = preds.squeeze()
                 target = y[:, -pred_len:, -1].squeeze()
            
            if preds.shape != target.shape:
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
            
            if preds.dim() == 3 and preds.shape[2] > 1:
                preds = preds[:, :, -1]
            
            preds = preds.squeeze().cpu().numpy()
            target = y[:, -pred_len:, -1].squeeze().cpu().numpy()
            
            if np.ndim(preds) == 0: preds = np.expand_dims(preds, axis=0)
            if np.ndim(target) == 0: target = np.expand_dims(target, axis=0)
                
            preds_list.extend(preds)
            trues_list.extend(target)
            
    target_col_idx = -1
    data_min = scaler.data_min_[target_col_idx]
    data_max = scaler.data_max_[target_col_idx]
    
    def manual_inverse(data_norm, min_val, max_val):
        return data_norm * (max_val - min_val) + min_val
    
    preds_inv = manual_inverse(np.array(preds_list), data_min, data_max)
    trues_inv = manual_inverse(np.array(trues_list), data_min, data_max)
    
    mse = mean_squared_error(trues_inv, preds_inv)
    mae = mean_absolute_error(trues_inv, preds_inv)
    r2 = r2_score(trues_inv, preds_inv)
    
    return {"MSE": mse, "MAE": mae, "R2": r2, "Preds": preds_inv, "Trues": trues_inv}

### 4. Main Execution ###

def main():
    # SETTINGS
    CAR_ID = 'V1' # <--- Change this to V2, V3 etc.
    root_path = '汇总/汇总'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data (Single Car Mode)
    print(f"Loading Data (Single Car: {CAR_ID})...")
    df_train, df_test = load_data_single_car(CAR_ID, 0.8, root_path)
    if df_train is None: return

    # Configuration
    WINDOW_SIZE = 9
    PRED_LEN = 1
    BATCH_SIZE = 32 # Smaller batch for single car due to less data
    INPUT_DIM = df_train.shape[1] - 1
    EPOCHS = 50
    LR = 0.001
    
    result_dir = 'experiment_results_single_car'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    print(f"Results will be saved to: {result_dir}")
    
    train_loader, test_loader, scaler = create_dataset_loader(df_train, df_test, WINDOW_SIZE, PRED_LEN, BATCH_SIZE)
    results = {}
    
    # Models
    models_to_run = [
        ("CNN-BiLSTM", CNNLSTM(INPUT_DIM, WINDOW_SIZE, 128, PRED_LEN)),
        ("Bi-GRU", BiGRUModel(INPUT_DIM, 128, 2, PRED_LEN)),
        ("CNN-BiGRU", CNNGRU(INPUT_DIM, WINDOW_SIZE, 128, PRED_LEN))
    ]
    
    for name, model in models_to_run:
        print(f"\nRunning {name}...")
        model = model.to(device)
        opt = optim.Adam(model.parameters(), lr=LR)
        crit = nn.MSELoss()
        train_model(model, train_loader, crit, opt, device, EPOCHS, PRED_LEN)
        results[name] = evaluate_model(model, test_loader, scaler, device, PRED_LEN)
        print(f"{name} Results: MSE={results[name]['MSE']:.4f}, R2={results[name]['R2']:.4f}")

    # LSTM + iTransformer
    print("\nRunning LSTM+Improved iTransformer...")
    class Config:
        def __init__(self):
            self.seq_len = WINDOW_SIZE
            self.label_len = int(WINDOW_SIZE / 2)
            self.pred_len = PRED_LEN
            self.freq = 'b'
            self.batch_size = BATCH_SIZE
            self.num_epochs = EPOCHS
            self.learning_rate = LR
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
            self.top_k = 5
            self.num_kernels = 6
            self.distil = 1
            self.embed = 'timeF'
            self.output_attention = 0
            self.task_name = 'short_term_forecast'
            self.moving_avg = WINDOW_SIZE - 1
            self.dropout = 0.1

    config = Config()
    model_itrans = iTransformer.Model(config).to(device)
    opt = optim.Adam(model_itrans.parameters(), lr=LR)
    crit = nn.MSELoss()
    train_model(model_itrans, train_loader, crit, opt, device, EPOCHS, PRED_LEN)
    results['LSTM_iTransformer_Improved'] = evaluate_model(model_itrans, test_loader, scaler, device, PRED_LEN)
    print(f"LSTM+iTransformer Results: MSE={results['LSTM_iTransformer_Improved']['MSE']:.4f}, R2={results['LSTM_iTransformer_Improved']['R2']:.4f}")
    
    # Save & Plot
    summary = []
    plot_data = pd.DataFrame()
    first_model = list(results.keys())[0]
    plot_data['True'] = results[first_model]['Trues']
    
    for name, res in results.items():
        summary.append({"Model": name, "MSE": res['MSE'], "MAE": res['MAE'], "R2": res['R2']})
        plot_data[name] = res['Preds']
        pd.DataFrame({"True": res['Trues'], "Pred": res['Preds']}).to_csv(os.path.join(result_dir, f"{CAR_ID}_{name}.csv"), index=False)
        
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(os.path.join(result_dir, f"{CAR_ID}_Summary.csv"), index=False)
    
    plt.figure(figsize=(15, 6))
    plot_len = min(200, len(plot_data))
    subset = plot_data.iloc[-plot_len:]
    plt.plot(subset.index, subset['True'], label='True Value', color='black', linewidth=2)
    for i, name in enumerate(results.keys()):
        plt.plot(subset.index, subset[name], label=name, alpha=0.7)
    plt.title(f'Model Comparison ({CAR_ID} - Last {plot_len} Steps)')
    plt.legend()
    plt.savefig(os.path.join(result_dir, f"{CAR_ID}_Plot.png"))
    
    print("\nExperiment Complete. Summary:")
    print(df_summary)

if __name__ == "__main__":
    main()
