import pathlib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

SERVICES_PATH = pathlib.Path(__file__).parent
MODEL_PATH = SERVICES_PATH / "models" / "modelo1.pth"

# Es necesario definir la arquitectura del modelo para poder cargar los pesos
class passengersRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, dropout=0.2):
        super(passengersRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def predict_demand(
    historical_data_path: str,
    days_to_predict: int,
    sequence_length: int = 45
) -> pd.DataFrame:
    """
    Carga un modelo LSTM entrenado y predice la demanda de pasajeros para un número de días futuros.

    Args:
        model_path (str): Ruta al archivo del modelo entrenado (.pth).
        historical_data_path (str): Ruta al archivo CSV con los datos históricos.
                                    Debe contener las columnas 'Month' y '#Passengers'.
        days_to_predict (int): Número de días a predecir en el futuro.
        sequence_length (int): Longitud de la secuencia utilizada para entrenar el modelo.

    Returns:
        pd.DataFrame: Un DataFrame con las fechas y las predicciones de demanda.
    """
    # Determinar el dispositivo (CPU o GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Cargar y preparar el modelo
    model = passengersRNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 2. Cargar y procesar los datos históricos
    try:
        df_hist = pd.read_csv(historical_data_path)
        df_hist['Month'] = pd.to_datetime(df_hist['Month'])
        df_hist.sort_values('Month', inplace=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo de datos históricos en: {historical_data_path}")

    # 3. Escalar los datos
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_hist['#Passengers'].values.reshape(-1, 1))

    # 4. Obtener la última secuencia de los datos históricos como punto de partida
    last_sequence_scaled = scaled_data[-sequence_length:]
    initial_sequence = torch.FloatTensor(last_sequence_scaled).reshape(1, sequence_length, 1).to(device)

    # 5. Generar predicciones de forma autorregresiva
    predictions_scaled = []
    current_sequence = initial_sequence.clone()

    with torch.no_grad():
        for _ in range(days_to_predict):
            pred = model(current_sequence)
            predictions_scaled.append(pred.item())
            # Actualizar la secuencia: eliminar el valor más antiguo y añadir la nueva predicción
            new_entry = pred.reshape(1, 1, 1)
            current_sequence = torch.cat((current_sequence[:, 1:, :], new_entry), dim=1)

    # 6. Invertir la escala de las predicciones para obtener los valores reales
    final_predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))

    # 7. Crear el DataFrame de resultados con fechas futuras
    last_date = df_hist['Month'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]

    results_df = pd.DataFrame({
        'Fecha': future_dates,
        'Demanda_Predicha': final_predictions.flatten().astype(int)
    })

    return results_df

# --- Ejemplo de uso ---

# try:
#     # Predecir la demanda para los próximos 30 días
#     future_demand = predict_demand(
#         model_path=r'C:\Users\Alejandro Feria\Desktop\Unal\7 semestre\redes_neuronales\RNAB\api\src\services\models\modelo1.pth',
#         historical_data_path='./ruta_location.csv',
#         days_to_predict=30
#     )
#     print("Predicciones de demanda para los próximos 30 días:")
#     print(future_demand)
# except Exception as e:
#     print(f"Ocurrió un error: {e}")

