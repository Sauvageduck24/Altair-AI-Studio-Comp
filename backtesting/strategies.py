"""
Estrategias diferebntes para el POC de altair
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class BaseStrategy(ABC):

    def __init__(self, name: str):
        self.name = name
        self.params = {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_param_ranges(self) -> Dict:
        pass

    def get_params(self) -> Dict:
        return self.params.copy()

    def set_params(self, **params):
        self.params.update(params)


class EMAStrategy(BaseStrategy):
    """Estrategia basada en media móvil exponencial"""

    def __init__(self, ema_period: int = 20):
        super().__init__("EMA")
        self.params = {'ema_period': ema_period}

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        ema_period = self.params['ema_period']

        # Calcular EMA
        df['ema'] = df['close'].ewm(span=ema_period).mean()

        # Detectar cruces
        df['below_ema'] = (df['close'] < df['ema']).fillna(False)
        df['above_ema'] = (df['close'] > df['ema']).fillna(False)

        df['prev_below'] = df['below_ema'].shift(1).fillna(False)
        df['prev_above'] = df['above_ema'].shift(1).fillna(False)

        df['cross_below'] = (~df['prev_below']) & df['below_ema']
        df['cross_above'] = (~df['prev_above']) & df['above_ema']

        # Señales
        df['signal'] = 0
        df.loc[df['cross_below'], 'signal'] = 1   # Compra
        df.loc[df['cross_above'], 'signal'] = -1  # Venta

        return df

    def get_param_ranges(self) -> Dict:
        return {'ema_period': (5, 100)}


class RSIStrategy(BaseStrategy):
    """Estrategia basada en RSI (Relative Strength Index)"""

    def __init__(self, rsi_period: int = 14, oversold: int = 30, overbought: int = 70):
        super().__init__("RSI")
        self.params = {
            'rsi_period': rsi_period,
            'oversold': oversold,
            'overbought': overbought
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        rsi_period = self.params['rsi_period']
        oversold = self.params['oversold']
        overbought = self.params['overbought']

        # Calcular RSI nativo
        df['rsi'] = self._calculate_rsi(df['close'], rsi_period)

        # Señales RSI
        df['signal'] = 0
        df.loc[df['rsi'] < oversold, 'signal'] = 1   # Compra en sobreventa
        df.loc[df['rsi'] > overbought, 'signal'] = -1  # Venta en sobrecompra

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calcula RSI de forma nativa"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def get_param_ranges(self) -> Dict:
        return {
            'rsi_period': (10, 30),
            'oversold': (20, 35),
            'overbought': (65, 80)
        }


class MACDStrategy(BaseStrategy):
    """Estrategia basada en MACD"""

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__("MACD")
        self.params = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        fast = self.params['fast_period']
        slow = self.params['slow_period']
        signal_period = self.params['signal_period']

        # Calcular MACD nativo
        macd_line, signal_line, histogram = self._calculate_macd(
            df['close'], fast, slow, signal_period)

        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = histogram

        # Señales basadas en cruce de MACD con línea de señal
        df['signal'] = 0
        df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & (
            df['macd'].shift(1) <= df['macd_signal'].shift(1))
        df['macd_cross_down'] = (df['macd'] < df['macd_signal']) & (
            df['macd'].shift(1) >= df['macd_signal'].shift(1))

        df.loc[df['macd_cross_up'], 'signal'] = 1   # Compra
        df.loc[df['macd_cross_down'], 'signal'] = -1  # Venta

        return df

    def _calculate_macd(self, prices: pd.Series, fast: int, slow: int, signal: int) -> tuple:
        """Calcula MACD de forma nativa"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def get_param_ranges(self) -> Dict:
        return {
            'fast_period': (8, 20),
            'slow_period': (20, 35),
            'signal_period': (5, 15)
        }


class BollingerStrategy(BaseStrategy):
    """Estrategia basada en Bandas de Bollinger"""

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0):
        super().__init__("Bollinger")
        self.params = {
            'bb_period': bb_period,
            'bb_std': bb_std
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        period = self.params['bb_period']
        std_dev = self.params['bb_std']

        # Calcular Bandas de Bollinger nativo
        upper, middle, lower = self._calculate_bollinger_bands(
            df['close'], period, std_dev)

        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower

        # Señales: compra cuando toca banda inferior, venta cuando toca superior
        df['signal'] = 0
        df.loc[df['close'] <= df['bb_lower'], 'signal'] = 1   # Compra
        df.loc[df['close'] >= df['bb_upper'], 'signal'] = -1  # Venta

        return df

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int, std_dev: float) -> tuple:
        """Calcula Bandas de Bollinger de forma nativa"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    def get_param_ranges(self) -> Dict:
        return {
            'bb_period': (15, 30),
            'bb_std': (1.5, 2.5)
        }


class StochasticStrategy(BaseStrategy):
    """Estrategia basada en Oscilador Estocástico"""

    def __init__(self, k_period: int = 14, d_period: int = 3, oversold: int = 20, overbought: int = 80):
        super().__init__("Stochastic")
        self.params = {
            'k_period': k_period,
            'd_period': d_period,
            'oversold': oversold,
            'overbought': overbought
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        k_period = self.params['k_period']
        d_period = self.params['d_period']
        oversold = self.params['oversold']
        overbought = self.params['overbought']

        # Calcular Estocástico nativo
        stoch_k, stoch_d = self._calculate_stochastic(
            df['high'], df['low'], df['close'], k_period, d_period)

        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d

        # Señales basadas en niveles de sobrecompra/sobreventa
        df['signal'] = 0
        df.loc[df['stoch_k'] < oversold, 'signal'] = 1   # Compra en sobreventa
        df.loc[df['stoch_k'] > overbought, 'signal'] = - \
            1  # Venta en sobrecompra

        return df

    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int, d_period: int) -> tuple:
        """Calcula Oscilador Estocástico de forma nativa"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()

        return k_percent, d_percent

    def get_param_ranges(self) -> Dict:
        return {
            'k_period': (10, 20),
            'd_period': (3, 7),
            'oversold': (15, 25),
            'overbought': (75, 85)
        }


class StrategyPool:
    def __init__(self):
        self.strategies = {
            'EMA': EMAStrategy(),
            'RSI': RSIStrategy(),
            'MACD': MACDStrategy(),
            'Bollinger': BollingerStrategy(),
            'Stochastic': StochasticStrategy()
        }

        # Inicialmente todas las estrategias están activas en todos los clusters
        self.cluster_weights = {
            'cluster_0': {name: 1.0 for name in self.strategies.keys()},
            'cluster_1': {name: 1.0 for name in self.strategies.keys()}
        }

    def get_strategy(self, name: str) -> BaseStrategy:
        return self.strategies.get(name)

    def get_all_strategies(self) -> Dict[str, BaseStrategy]:
        return self.strategies

    def set_cluster_weights(self, cluster: str, weights: Dict[str, float]):
        if cluster not in self.cluster_weights:
            self.cluster_weights[cluster] = {}
        self.cluster_weights[cluster].update(weights)

    def get_active_strategies_for_cluster(self, cluster: str) -> List[str]:
        """estrategias con peso mayor a 0"""
        if cluster not in self.cluster_weights:
            return list(self.strategies.keys())

        return [name for name, weight in self.cluster_weights[cluster].items() if weight > 0]

    def generate_combined_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['combined_signal'] = 0

        for cluster in df['cluster'].unique():
            cluster_mask = df['cluster'] == cluster
            cluster_data = df[cluster_mask].copy()

            active_strategies = self.get_active_strategies_for_cluster(cluster)

            if not active_strategies:
                continue

            signals = []
            weights = []

            for strategy_name in active_strategies:
                strategy = self.strategies[strategy_name]
                weight = self.cluster_weights.get(
                    cluster, {}).get(strategy_name, 0.0)

                if weight > 0:
                    strategy_signals = strategy.generate_signals(cluster_data)
                    signals.append(strategy_signals['signal'] * weight)
                    weights.append(weight)

            if signals:
                combined = sum(signals) / \
                    sum(weights) if sum(weights) > 0 else 0
                combined_discrete = np.where(
                    combined > 0.5, 1, np.where(combined < -0.5, -1, 0))
                df.loc[cluster_mask, 'combined_signal'] = combined_discrete

        return df
