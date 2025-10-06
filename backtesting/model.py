import pandas as pd
import numpy as np


class EMAModel:
    def __init__(self, ema_period: int = 20):
        """
        Modelo simple de EMA que genera señales de compra/venta.
        
        Args:
            ema_period: Periodo para el cálculo de la EMA
        """
        self.ema_period = ema_period
    
    def calculate_ema(self, prices: pd.Series) -> pd.Series:
        """Calcula la EMA exponencial móvil"""
        return prices.ewm(span=self.ema_period).mean()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Genera señales de trading basadas en el cruce de precio con EMA.
        
        Compra cuando el precio cruza por debajo de la EMA (esperando rebote)
        Vende cuando el precio cruza por encima de la EMA (esperando caída)
        
        Args:
            data: DataFrame con columna 'close'
            
        Returns:
            DataFrame con columnas adicionales: 'ema', 'signal', 'position'
        """
        df = data.copy()
        
        # Calcular EMA
        df['ema'] = self.calculate_ema(df['close'])
        
        # Crear indicadores de posición relativa (asegurar tipo boolean)
        df['below_ema'] = (df['close'] < df['ema']).fillna(False)
        df['above_ema'] = (df['close'] > df['ema']).fillna(False)
        
        # Detectar cruces de forma más robusta
        df['prev_below'] = df['below_ema'].shift(1).fillna(False)
        df['prev_above'] = df['above_ema'].shift(1).fillna(False)
        
        df['cross_below'] = (~df['prev_below']) & df['below_ema']  # Precio cruza hacia abajo
        df['cross_above'] = (~df['prev_above']) & df['above_ema']  # Precio cruza hacia arriba
        
        # Generar señales basadas en cruces
        df['signal'] = 0
        df.loc[df['cross_below'], 'signal'] = 1   # Compra cuando cruza hacia abajo
        df.loc[df['cross_above'], 'signal'] = -1  # Vende cuando cruza hacia arriba
        
        return df
    
    def get_params(self) -> dict:
        """Retorna los parámetros del modelo"""
        return {'ema_period': self.ema_period}
    
    def set_params(self, **params):
        """Establece los parámetros del modelo"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
