"""
optimiza cada estrategia en su cluster específico y evalúa el rendimiento combinado.
"""
import pandas as pd
import numpy as np
import optuna
import vectorbt as vbt
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple
import warnings

from strategies import StrategyPool, BaseStrategy
from utils import load_data, shift_signals, force_close_open_positions_numba, filter_outlier_entries
warnings.simplefilter(action='ignore', category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ClusterOptimizer:
    def __init__(self, data: pd.DataFrame, strategy_pool: StrategyPool, use_outlier_filter: bool = True):
        self.data = data
        self.strategy_pool = strategy_pool
        self.use_outlier_filter = use_outlier_filter
        self.cluster_best_params = {}
        self.cluster_best_strategies = {}

    def optimize_cluster_strategy(self, cluster: str, strategy_name: str, n_trials: int = 50) -> Dict:
        print(f"\n=== Optimizando {strategy_name} para {cluster} ===")
        cluster_data = self.data[self.data['cluster'] == cluster].copy()
        print(f"Datos del cluster: {len(cluster_data)} registros")

        if len(cluster_data) < 100:
            print(f"Insuficientes datos para {cluster}")
            return None

        # pillar estrategia
        strategy = self.strategy_pool.get_strategy(strategy_name)
        if not strategy:
            print(f"Estrategia {strategy_name} no encontrada")
            return None

        study = optuna.create_study(
            direction='maximize',
            study_name=f'{strategy_name}_{cluster}_optimization'
        )

        def objective(trial):
            try:
                param_ranges = strategy.get_param_ranges()

                trial_params = {}
                for param_name, (min_val, max_val) in param_ranges.items():
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        trial_params[param_name] = trial.suggest_int(
                            param_name, min_val, max_val)
                    else:
                        trial_params[param_name] = trial.suggest_float(
                            param_name, min_val, max_val)

                # configurar estrategia con parámetros del trial
                strategy.set_params(**trial_params)

                data_with_signals = strategy.generate_signals(cluster_data)

                shifted_signals = shift_signals(data_with_signals['signal'])

                entries = shifted_signals == 1
                exits = shifted_signals == -1

                entries, exits = force_close_open_positions_numba(
                    entries.values, exits.values)

                entries = pd.Series(entries, index=cluster_data.index)
                exits = pd.Series(exits, index=cluster_data.index)

                # Filtrar outliers si está habilitado
                if self.use_outlier_filter:
                    entries = filter_outlier_entries(
                        entries, cluster_data['outlier_flag'])

                # Backtesting
                portfolio = vbt.Portfolio.from_signals(
                    cluster_data['close'],
                    entries=entries,
                    exits=exits,
                    init_cash=10000
                )

                total_return = portfolio.total_return()
                max_drawdown = abs(portfolio.max_drawdown())

                # objetivo rendimiento - drawdown
                objective_value = total_return - max_drawdown

                return objective_value

            except Exception as e:
                print(f"Error en trial {trial.number}: {e}")
                return -999

        # optimización
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        result = {
            'cluster': cluster,
            'strategy': strategy_name,
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': n_trials
        }

        print(
            f"Mejor valor para {strategy_name} en {cluster}: {study.best_value:.4f}")
        print(f"Mejores parámetros: {study.best_params}")

        return result

    def optimize_all_strategies_for_cluster(self, cluster: str, n_trials: int = 50) -> Dict:
        """
        Optimiza todas las estrategias para un cluster específico.

        Args:
            cluster: Nombre del cluster
            n_trials: Número de trials por estrategia

        Returns:
            Dict con resultados de todas las estrategias
        """
        results = {}

        for strategy_name in self.strategy_pool.get_all_strategies().keys():
            result = self.optimize_cluster_strategy(
                cluster, strategy_name, n_trials)
            if result:
                results[strategy_name] = result

        return results

    def find_best_strategy_per_cluster(self, n_trials: int = 50) -> Dict:
        all_results = {}

        clusters = self.data['cluster'].unique()

        for cluster in clusters:
            print(f"\n{'='*50}")
            print(f"OPTIMIZANDO CLUSTER: {cluster}")
            print(f"{'='*50}")

            cluster_results = self.optimize_all_strategies_for_cluster(
                cluster, n_trials)

            if cluster_results:
                # la mejor estrategia para este cluster
                best_strategy_name = max(cluster_results.keys(),
                                         key=lambda x: cluster_results[x]['best_value'])
                best_result = cluster_results[best_strategy_name]

                all_results[cluster] = {
                    'best_strategy': best_strategy_name,
                    'best_params': best_result['best_params'],
                    'best_value': best_result['best_value'],
                    'all_results': cluster_results
                }

                print(
                    f"\n*** MEJOR ESTRATEGIA PARA {cluster}: {best_strategy_name} ***")
                print(f"Valor: {best_result['best_value']:.4f}")
                print(f"Parámetros: {best_result['best_params']}")

        return all_results

    def configure_strategy_weights(self, optimization_results: Dict) -> Dict:
        """
        Configura los pesos de las estrategias basado en los resultados de optimización.
        Solo la mejor estrategia por cluster tendrá peso 1.0, el resto 0.0.

        Args:
            optimization_results: Resultados de optimización por cluster

        Returns:
            Dict con configuración de pesos
        """
        weight_config = {}

        for cluster, cluster_info in optimization_results.items():
            best_strategy = cluster_info['best_strategy']

            # Crear configuración de pesos para este cluster
            cluster_weights = {}
            for strategy_name in self.strategy_pool.get_all_strategies().keys():
                cluster_weights[strategy_name] = 1.0 if strategy_name == best_strategy else 0.0

            weight_config[cluster] = {
                'weights': cluster_weights,
                'best_strategy': best_strategy,
                'best_params': cluster_info['best_params']
            }

        return weight_config

    def evaluate_combined_performance(self, weight_config: Dict) -> Dict:
        """
        Evalúa el rendimiento del sistema combinado con los pesos optimizados.

        Args:
            weight_config: Configuración de pesos por cluster

        Returns:
            Dict con métricas de rendimiento
        """
        print(f"\n{'='*50}")
        print("EVALUANDO RENDIMIENTO COMBINADO")
        print(f"{'='*50}")

        for cluster, config in weight_config.items():
            self.strategy_pool.set_cluster_weights(cluster, config['weights'])
            best_strategy_name = config['best_strategy']
            best_strategy = self.strategy_pool.get_strategy(best_strategy_name)
            if best_strategy:
                best_strategy.set_params(**config['best_params'])

        combined_data = self.strategy_pool.generate_combined_signals(self.data)

        shifted_signals = shift_signals(combined_data['combined_signal'])

        entries = shifted_signals == 1
        exits = shifted_signals == -1

        entries, exits = force_close_open_positions_numba(
            entries.values, exits.values)

        entries = pd.Series(entries, index=self.data.index)
        exits = pd.Series(exits, index=self.data.index)

        # Filtrar outliers si está habilitado
        if self.use_outlier_filter:
            entries = filter_outlier_entries(
                entries, self.data['outlier_flag'])

        portfolio = vbt.Portfolio.from_signals(
            self.data['close'],
            entries=entries,
            exits=exits,
            init_cash=1_000_000
        )

        # métricas
        total_return = portfolio.total_return()
        max_drawdown = abs(portfolio.max_drawdown())

        try:
            vbt.settings.array_wrapper['freq'] = '15min'
            sharpe = portfolio.sharpe_ratio()
        except:
            sharpe = 0.0

        try:
            num_trades = len(portfolio.trades.records_readable)
            trades_pnl = portfolio.trades.pnl.values
            win_rate = (trades_pnl > 0).mean() if len(trades_pnl) > 0 else 0.0
        except:
            num_trades = 0
            win_rate = 0.0

        # resultados por cluster
        for cluster in self.data['cluster'].unique():
            cluster_mask = combined_data['cluster'] == cluster
            cluster_signals = combined_data.loc[cluster_mask,
                                                'combined_signal']
            cluster_entries = entries[cluster_mask]
            cluster_exits = exits[cluster_mask]

            print(f"\n{cluster}:")
            print(
                f"  Estrategia activa: {weight_config[cluster]['best_strategy']}")
            print(f"  Señales generadas: {abs(cluster_signals).sum()}")
            print(f"  Entradas: {cluster_entries.sum()}")
            print(f"  Salidas: {cluster_exits.sum()}")

        results = {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'total_trades': num_trades,
            'win_rate': win_rate,
            'objective_value': total_return - max_drawdown,
            'total_entries': entries.sum(),
            'total_exits': exits.sum(),
            'weight_config': weight_config
        }

        print(f"\n=== MÉTRICAS FINALES ===")
        print(f"Rendimiento total: {total_return*100:.4f}%")
        print(f"Max Drawdown: {max_drawdown*100:.4f}%")
        print(f"Sharpe Ratio: {sharpe:.4f}")
        print(f"Número de trades: {num_trades}")
        print(f"WIN Rate: {win_rate:.4f}")
        print(f"Métrica objetivo: {results['objective_value']:.4f}")
        print(f"Total entradas: {entries.sum()}")
        print(f"Total salidas: {exits.sum()}")

        return results


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Optimización multi-estrategia por clusters')
    parser.add_argument('--use-outlier-filter', action='store_true', default=True,
                        help='Usar filtro de anomalías')
    parser.add_argument('--n-trials', type=int, default=30,
                        help='Número de trials por estrategia (default: 30)')
    parser.add_argument('--sample-size', type=int, default=50000,
                        help='Tamaño de muestra para optimización (default: 50000)')

    return parser.parse_args()


def main():
    args = parse_arguments()

    print(f"=== CONFIGURACIÓN ===")
    print(
        f"Filtro de anomalías: {'ACTIVADO' if args.use_outlier_filter else 'DESACTIVADO'}")
    print(f"Número de trials: {args.n_trials}")
    print(f"Tamaño de muestra: {args.sample_size}")
    print("=" * 20)

    data_path = Path(__file__).parent.parent / \
        "data_preprocessed" / "enhanced_eurusd_dataset.csv"
    data = load_data(str(data_path))

    print(
        f"Datos cargados: {len(data)} registros desde {data.index[0]} hasta {data.index[-1]}")

    sample_size = min(args.sample_size, len(data))
    data_sample = data.iloc[-sample_size:].copy()
    print(f"Usando muestra de {len(data_sample)} registros para optimización")

    strategy_pool = StrategyPool()
    optimizer = ClusterOptimizer(
        data_sample, strategy_pool, args.use_outlier_filter)
    optimization_results = optimizer.find_best_strategy_per_cluster(
        args.n_trials)
    weight_config = optimizer.configure_strategy_weights(optimization_results)
    combined_results = optimizer.evaluate_combined_performance(weight_config)

    output_data = {
        'optimization_results': optimization_results,
        'weight_config': weight_config,
        'combined_performance': combined_results,
        'configuration': {
            'use_outlier_filter': args.use_outlier_filter,
            'n_trials': args.n_trials,
            'sample_size': sample_size
        }
    }

    # Guardar en JSON arreglado
    filename = "cluster_optimization_results_filtered.json" if args.use_outlier_filter else "cluster_optimization_results_original.json"
    results_path = Path(__file__).parent / filename

    with open(results_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResultados guardados en: {results_path}")


if __name__ == "__main__":
    main()
