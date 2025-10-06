import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit

def load_data(file_path: str) -> pd.DataFrame:
    """
    Carga los datos del dataset preprocesado.
    
    Args:
        file_path: Ruta al archivo CSV
        
    Returns:
        DataFrame con los datos cargados y datetime index
    """
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df


def shift_signals(signals: pd.Series, shift_periods: int = 1) -> pd.Series:
    """
    Desplaza las señales hacia adelante para simular ejecución en la siguiente vela.
    
    Args:
        signals: Serie con las señales de trading
        shift_periods: Número de periodos a desplazar (default: 1)
        
    Returns:
        Serie con señales desplazadas
    """
    shifted_signals = signals.shift(shift_periods)
    # Rellenar los primeros valores con 0 (sin señal)
    shifted_signals.fillna(0, inplace=True)
    return shifted_signals


@njit(fastmath=True, cache=False)
def force_close_open_positions_numba(entries_arr, exits_arr):
    n = len(entries_arr)
    if n == 0:
        return entries_arr, exits_arr

    entries = entries_arr.copy()
    exits = exits_arr.copy()

    open_pos = False
    for i in range(n):
        # Priorizar salida si coinciden
        if entries[i] and exits[i]:
            entries[i] = False  # mantenemos salida

        if open_pos:
            # bloquear nuevas entradas hasta cerrar
            if entries[i]:
                entries[i] = False
            if exits[i]:
                open_pos = False
        else:
            # bloquear salidas sin posición
            if exits[i]:
                exits[i] = False
            if entries[i]:
                open_pos = True

    return entries, exits


def filter_outlier_entries(entries: pd.Series, outlier_flags: pd.Series) -> pd.Series:
    """
    Anula las entradas donde outlier_flag es "Outlier".
    
    Args:
        entries: Serie booleana con las señales de entrada
        outlier_flags: Serie con los flags de outliers
        
    Returns:
        Serie de entradas filtrada sin outliers
    """
    # Crear una copia de las entradas
    filtered_entries = entries.copy()
    
    # Anular entradas donde hay outliers
    outlier_mask = outlier_flags == "Outlier"
    filtered_entries[outlier_mask] = False
    
    return filtered_entries