import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

def initialize_mt5():
    """Inicializa la conexión con MetaTrader 5"""
    if not mt5.initialize():
        print("Error al inicializar MetaTrader 5")
        print("Error:", mt5.last_error())
        return False
    return True

def download_eurusd_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M15, start_date=None, end_date=None):
    """
    Descarga datos históricos de EURUSD desde MetaTrader 5
    
    Args:
        symbol (str): Símbolo de la divisa (por defecto EURUSD)
        timeframe: Marco temporal (por defecto 15 minutos)
        start_date (datetime): Fecha de inicio para la descarga
        end_date (datetime): Fecha de fin para la descarga
    
    Returns:
        pd.DataFrame: DataFrame con los datos descargados
    """
    
    # Usar fechas por defecto si no se proporcionan
    if start_date is None:
        start_date = datetime(2010, 1, 1)
    if end_date is None:
        end_date = datetime(2020, 1, 1)
    
    print(f"Descargando datos de {symbol} desde {start_date.strftime('%Y-%m-%d')} hasta {end_date.strftime('%Y-%m-%d')}")
    
    # Obtener datos históricos
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    
    if rates is None:
        print(f"Error al descargar datos para {symbol}")
        print("Error:", mt5.last_error())
        return None
    
    # Convertir a DataFrame
    df = pd.DataFrame(rates)
    
    # Convertir timestamp a datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Renombrar columnas para mayor claridad
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
    
    print(f"Datos descargados exitosamente: {len(df)} registros")
    print(f"Rango de fechas: {df['timestamp'].min()} a {df['timestamp'].max()}")
    
    return df

def save_data(df, filename="EURUSD_15M.csv", folder="data_raw"):
    """
    Guarda los datos en un archivo CSV
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        filename (str): Nombre del archivo
        folder (str): Carpeta donde guardar el archivo
    """
    
    # Crear carpeta si no existe
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Carpeta '{folder}' creada")
    
    # Ruta completa del archivo
    filepath = os.path.join(folder, filename)
    
    # Guardar datos
    df.to_csv(filepath, index=False)
    print(f"Datos guardados en: {filepath}")
    
    return filepath

def get_symbol_info(symbol="EURUSD"):
    """Obtiene información del símbolo"""
    symbol_info = mt5.symbol_info(symbol)
    
    if symbol_info is None:
        print(f"Símbolo {symbol} no encontrado")
        return None
    
    print(f"\nInformación del símbolo {symbol}:")
    print(f"Descripción: {symbol_info.description}")
    print(f"Spread: {symbol_info.spread}")
    print(f"Dígitos: {symbol_info.digits}")
    print(f"Punto: {symbol_info.point}")
    
    return symbol_info

def main():
    """Función principal"""
    print("=== Descargador de datos MetaTrader 5 ===")
    print("Símbolo: EURUSD")
    print("Marco temporal: 15 minutos")
    print("Período: 2010-01-01 hasta 2020-01-01")
    
    # Inicializar MT5
    if not initialize_mt5():
        sys.exit(1)
    
    try:
        # Obtener información del símbolo
        get_symbol_info("EURUSD")
        
        # Configuración de descarga
        symbol = "EURUSD"
        timeframe = mt5.TIMEFRAME_M15
        start_date = datetime(2010, 1, 1)  # Fecha de inicio: 1 de enero de 2010
        end_date = datetime(2020, 1, 1)    # Fecha de fin: 1 de enero de 2020
        
        # Descargar datos
        df = download_eurusd_data(symbol, timeframe, start_date, end_date)
        
        # cambiale el nombre a la columna timestamp por time
        df.rename(columns={'timestamp': 'time'}, inplace=True)
        
        if df is not None:
            # Mostrar resumen de datos
            print(f"\nResumen de datos descargados:")
            print(f"Total de registros: {len(df)}")
            print(f"Primer registro: {df['time'].iloc[0]}")
            print(f"Último registro: {df['time'].iloc[-1]}")
            print(f"Precio de apertura inicial: {df['open'].iloc[0]}")
            print(f"Precio de cierre final: {df['close'].iloc[-1]}")
            
            # Mostrar primeras filas
            print(f"\nPrimeras 5 filas:")
            print(df.head())
            
            # Guardar datos
            filepath = save_data(df)
            
            # Mostrar estadísticas básicas
            print(f"\nEstadísticas básicas:")
            print(df[['open', 'high', 'low', 'close', 'tick_volume']].describe())
            
        else:
            print("No se pudieron descargar los datos")
            
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        
    finally:
        # Cerrar conexión con MT5
        mt5.shutdown()
        print("\nConexión con MetaTrader 5 cerrada")

if __name__ == "__main__":
    main()
