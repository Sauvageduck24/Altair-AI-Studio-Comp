import pandas as pd
import numpy as np
import optuna
import vectorbt as vbt
from pathlib import Path
import argparse

from model import EMAModel
from utils import load_data, shift_signals, force_close_open_positions_numba, filter_outlier_entries
import warnings

# Ignorar FutureWarnings de pandas y vectorbt
warnings.simplefilter(action='ignore', category=FutureWarning)

# evitar mensajes optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, use_outlier_filter=True):
    """
    Función objetivo para optimización con Optuna.
    
    Args:
        trial: Trial de Optuna
        use_outlier_filter: Si True, aplica el filtro de outliers
        
    Returns:
        Métrica a optimizar (rendimiento final - max drawdown)
    """
    # Parámetros a optimizar
    ema_period = trial.suggest_int('ema_period', 1, 200)
    
    # Crear modelo con parámetros del trial
    model = EMAModel(ema_period=ema_period)
    
    # Generar señales
    data_with_signals = model.generate_signals(data)
    
    # Desplazar señales para simular ejecución en siguiente vela
    shifted_signals = shift_signals(data_with_signals['signal'])
    
    # Convertir señales a entradas/salidas para vectorbt
    entries = shifted_signals == 1
    exits = shifted_signals == -1
    
    # Forzar cierre de posiciones abiertas para evitar solapamientos
    entries, exits = force_close_open_positions_numba(entries.values, exits.values)
    
    # Convertir de numpy arrays de vuelta a pandas Series para mantener índices
    entries = pd.Series(entries, index=data.index)
    exits = pd.Series(exits, index=data.index)
    
    # Filtrar entradas en outliers solo si está habilitado
    if use_outlier_filter:
        entries = filter_outlier_entries(entries, data['outlier_flag'])
    
    try:
        # Ejecutar backtesting con vectorbt
        portfolio = vbt.Portfolio.from_signals(
            data['close'],
            entries=entries,
            exits=exits,
            init_cash=10000
        )
        
        # Calcular métricas
        total_return = portfolio.total_return()
        max_drawdown = abs(portfolio.max_drawdown())  # Asegurar valor positivo
        
        # Métrica objetivo: rendimiento final - max drawdown
        objective_value = total_return - max_drawdown
        
        return objective_value
        
    except Exception as e:
        # Si hay error en el backtesting, retornar valor muy bajo
        print(f"Error en trial {trial.number}: {e}")
        return -999


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Optimización del modelo EMA con opción de filtro de anomalías')
    parser.add_argument('--use-outlier-filter', action='store_true', default=True,
                        help='Usar filtro de anomalías (filter_outlier_entries)')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Número de trials para la optimización (default: 50)')
    parser.add_argument('--sample-size', type=int, default=500000,
                        help='Tamaño de muestra para optimización (default: 50000)')
    
    return parser.parse_args()


def main():
    """Función principal para ejecutar la optimización"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    print(f"=== CONFIGURACIÓN ===")
    print(f"Filtro de anomalías: {'ACTIVADO' if args.use_outlier_filter else 'DESACTIVADO'}")
    print(f"Número de trials: {args.n_trials}")
    print(f"Tamaño de muestra: {args.sample_size}")
    print("=" * 20)
    
    # Cargar datos
    data_path = Path(__file__).parent.parent / "data_preprocessed" / "enhanced_eurusd_dataset.csv"
    global data
    data = load_data(str(data_path))
    
    print(f"Datos cargados: {len(data)} registros desde {data.index[0]} hasta {data.index[-1]}")
    
    # Usar solo una muestra para hacer la optimización más rápida
    # Puedes cambiar esto para usar todos los datos
    sample_size = min(args.sample_size, len(data))
    data = data.iloc[-sample_size:].copy()
    print(f"Usando muestra de {len(data)} registros para optimización")
    
    # Crear estudio de Optuna
    study = optuna.create_study(
        direction='maximize',
        study_name='ema_optimization'
    )
    
    print("Iniciando optimización...")
    
    # Ejecutar optimización
    study.optimize(lambda trial: objective(trial, use_outlier_filter=args.use_outlier_filter), 
                   n_trials=args.n_trials, show_progress_bar=True)
    
    # Mostrar resultados
    print("\n=== RESULTADOS DE OPTIMIZACIÓN ===")
    print(f"Mejor valor objetivo: {study.best_value:.4f}")
    print(f"Mejores parámetros: {study.best_params}")
    
    # Probar el mejor modelo
    print("\n=== EVALUACIÓN DEL MEJOR MODELO ===")
    best_model = EMAModel(**study.best_params)
    
    # Generar señales con el mejor modelo
    final_data = best_model.generate_signals(data)
    shifted_signals = shift_signals(final_data['signal'])
    
    entries = shifted_signals == 1
    exits = shifted_signals == -1
    
    # Forzar cierre de posiciones abiertas para evitar solapamientos
    entries, exits = force_close_open_positions_numba(entries.values, exits.values)
    
    # Convertir de numpy arrays de vuelta a pandas Series para mantener índices
    entries = pd.Series(entries, index=data.index)
    exits = pd.Series(exits, index=data.index)
    
    # Filtrar entradas en outliers solo si está habilitado
    if args.use_outlier_filter:
        entries = filter_outlier_entries(entries, data['outlier_flag'])
        print("Filtro de anomalías aplicado en evaluación final")
    else:
        print("Filtro de anomalías NO aplicado en evaluación final")
    
    # Backtesting final
    portfolio = vbt.Portfolio.from_signals(
        data['close'],
        entries=entries,
        exits=exits,
        init_cash=1_000_000
    )
    
    # mostrar métrica final
    print(f"Métrica final (Rendimiento - Max Drawdown): {portfolio.total_return() - abs(portfolio.max_drawdown()):.4f}")

    # Mostrar estadísticas finales
    print(f"Rendimiento total: {portfolio.total_return()*100:.4f} %")
    print(f"Max Drawdown: {abs(portfolio.max_drawdown())*100:.4f} %")
    
    # Calcular Sharpe ratio de forma segura
    try:
        # Configurar settings de vectorbt para evitar el error de frecuencia
        vbt.settings.array_wrapper['freq'] = '15min'
        sharpe = portfolio.sharpe_ratio()
        print(f"Sharpe Ratio: {sharpe:.4f}")
    except Exception as e:
        sharpe = 0.0
        print(f"Sharpe Ratio: No calculable - {str(e)}")
    
    # Calcular número de trades de forma segura
    try:
        if hasattr(portfolio, 'trades'):
            num_trades = len(portfolio.trades.records_readable)
            print(f"Número de trades: {num_trades}")
        else:
            num_trades = 0
            print(f"Número de trades: N/A")
    except Exception as e:
        num_trades = 0
        print(f"Número de trades: Error - {str(e)}")
    
    # Calcular win rate de forma segura
    try:
        if hasattr(portfolio, 'trades') and len(portfolio.trades.records_readable) > 0:
            # Usar el método correcto de vectorbt para obtener PnL
            trades_pnl = portfolio.trades.pnl.values
            win_rate = (trades_pnl > 0).mean()
            print(f"Win Rate: {win_rate:.4f}")
        else:
            win_rate = 0.0
            print(f"Win Rate: No calculable (sin trades)")
    except Exception as e:
        win_rate = 0.0
        print(f"Win Rate: No calculable - {str(e)}")
    
    # Mostrar información adicional para debug
    print(f"Número de señales de compra: {entries.sum()}")
    print(f"Número de señales de venta: {exits.sum()}")
    
    # Guardar resultados
    results_df = pd.DataFrame([{
        'ema_period': study.best_params['ema_period'],
        'use_outlier_filter': args.use_outlier_filter,
        'total_return': portfolio.total_return(),
        'max_drawdown': abs(portfolio.max_drawdown()),
        'sharpe_ratio': sharpe,
        'total_trades': num_trades,
        'win_rate': win_rate,
        'objective_value': study.best_value
    }])
    
    # Nombrar el archivo según si se usa el filtro de anomalías o no
    filename = "optimization_results_filtered.csv" if args.use_outlier_filter else "optimization_results_original.csv"
    results_path = Path(__file__).parent.parent / "backtesting" / filename
    results_df.to_csv(results_path, index=False)
    print(f"\nResultados guardados en: {results_path}")


if __name__ == "__main__":
    main()
