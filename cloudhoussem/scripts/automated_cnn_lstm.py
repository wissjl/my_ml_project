# %%
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, MaxPooling1D, Flatten
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler

# %%


# %%


# %%


# %%


# %%


# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, TimeDistributed, Flatten
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def prepare_data_cnn_lstm_vol(df_returns, return_col='BTC_returns', vol_col='Realized_Volatility', 
                            n_lookback=60, n_steps=1, test_size=0.2):
    """
    Prepares 4D data for CNN-LSTM models using returns and realized volatility.
    """
    # Extract and reshape data
    data_x = df_returns[return_col].values.reshape(-1, 1)
    data_y = df_returns[vol_col].values.reshape(-1, 1)
    timestamps = df_returns.index

    # Scale data
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaled_x = scaler_x.fit_transform(data_x)
    scaled_y = scaler_y.fit_transform(data_y)

    # Create sequences
    X, y, time_index = [], [], []
    for i in range(n_lookback, len(scaled_x)):
        X.append(scaled_x[i - n_lookback:i])
        y.append(scaled_y[i])
        time_index.append(timestamps[i])

    # Convert to arrays and reshape for CNN-LSTM (samples, timesteps, features, channels)
    X = np.array(X).reshape(-1, n_steps, n_lookback, 1)
    y = np.array(y)
    time_index = np.array(time_index)

    # Split train/test
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    plot_dates = time_index[split_idx:]

    print(f"Prepared shapes -> X: {X.shape}, y: {y.shape}, Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test, plot_dates, scaler_y

def build_cnn_lstm(u1, u2, input_shape):
    """Builds a CNN-LSTM model with proper dimensions."""
    model = Sequential()
    
    # CNN layers - process each timestep independently
    model.add(TimeDistributed(
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),  # Added padding
        input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    
    # LSTM layers
    model.add(LSTM(u1, return_sequences=True))
    model.add(LSTM(u2))
    
    # Output layer
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    return model

def evaluate_model(y_true, y_pred, scaler):
    """Evaluates model performance with multiple metrics."""
    # Inverse transform predictions
    y_true = scaler.inverse_transform(y_true.reshape(-1, 1))
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
    
    metrics = {
        'R2': r2_score(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred)
    }
    return metrics

# Main execution
def run_cnn_lstm_model(input_file, output_base_path):
    os.makedirs(os.path.dirname(output_base_path), exist_ok=True)

    try:
        df = pd.read_csv(input_file, delimiter=';')
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return

    # Date parsing
    df['timestamp'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M', dayfirst=True)
    df.set_index('timestamp', inplace=True)
    df = df[['BTC']].sort_index().dropna()

    # Feature engineering
    df_returns = np.log(df / df.shift(1)).dropna()
    df_returns.rename(columns={'BTC': 'BTC_returns'}, inplace=True)
    df_returns['Realized_Volatility'] = df_returns['BTC_returns'].rolling(window=20).std()
    df_returns = df_returns.dropna()

    # Data prep
    X_train, X_test, y_train, y_test, plot_dates, scaler_y = prepare_data_cnn_lstm_vol(
        df_returns, n_lookback=60, test_size=0.2)

    layer_sizes = [2**i for i in range(1, 7)]  # [2, 4, 8, ..., 64]
    results = []

    for u1 in layer_sizes:
        for u2 in layer_sizes:
            try:
                print(f"\nTraining CNN-LSTM with layers ({u1}, {u2})")
                model = build_cnn_lstm(u1, u2, X_train.shape[1:])
                model.fit(X_train, y_train,
                          epochs=30,
                          batch_size=32,
                          validation_data=(X_test, y_test),
                          verbose=0)
                y_pred = model.predict(X_test)
                metrics = evaluate_model(y_test, y_pred, scaler_y)
                results.append({
                    'LSTM1': u1, 'LSTM2': u2,
                    **metrics
                })
            except Exception as e:
                print(f"⚠️ Failed ({u1}, {u2}): {str(e)}")
                results.append({'LSTM1': u1, 'LSTM2': u2, 'R2': None, 'MSE': None, 'RMSE': None, 'MAE': None})

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_base_path + ".csv", index=False)

    # Plotting
    metrics = ['R2', 'MSE', 'RMSE', 'MAE']
    titles = ['R² Score', 'Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error']
    colors = ['b', 'r', 'g', 'm']

    fig = plt.figure(figsize=(20, 16))
    for i, (metric, title, color) in enumerate(zip(metrics, titles, colors), 1):
        ax = fig.add_subplot(2, 2, i, projection='3d')
        pivot_df = results_df.pivot(index='LSTM1', columns='LSTM2', values=metric)
        X, Y = np.meshgrid(pivot_df.columns, pivot_df.index)
        Z = pivot_df.values
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.8)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        ax.set_xlabel('LSTM2 Units')
        ax.set_ylabel('LSTM1 Units')
        ax.set_zlabel(title)
        ax.set_title(f'{title}')
        ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.savefig(output_base_path + ".png")
    plt.close()



