# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Backtesting imports

import vectorbt as vbt
from pathlib import Path
import sys
    
from backtesting.model import EMAModel
from backtesting.utils import load_data as load_backtest_data, shift_signals, force_close_open_positions_numba, filter_outlier_entries
from backtesting.strategies import EMAStrategy, RSIStrategy, MACDStrategy, BollingerStrategy, StochasticStrategy
BACKTESTING_AVAILABLE = True

# Page configuration
st.set_page_config(
    page_title="EUR/USD Trading Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stSelectbox > div > div > select {
        background-color: #f0f2f6;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .cluster-info {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and prepare data with cache for performance optimization"""
    try:
        # Load raw OHLC data
        raw = pd.read_csv('data_raw/EURUSD_15M.csv')
        if 'time' not in raw.columns:
            st.error("The candle CSV must have a 'time' column.")
            return None
        
        raw['time'] = pd.to_datetime(raw['time'])
        for c in ['open','high','low','close']:
            raw[c] = pd.to_numeric(raw[c], errors='coerce')

        # Load preprocessed data
        prep = pd.read_csv('data_preprocessed/results.csv', sep=';')
        
        # If it has time column, convert it
        if 'time' in prep.columns:
            prep['time'] = pd.to_datetime(prep['time'])

        # Filter necessary columns
        columns_needed = ['cluster', 'outlier_flag']
        if 'time' in prep.columns:
            columns_needed.append('time')
        
        prep_filtered = prep[[col for col in columns_needed if col in prep.columns]].copy()

        # Data merge
        if 'time' in prep_filtered.columns:
            df = pd.merge(raw, prep_filtered, on='time', how='inner')
        else:
            raw = raw.reset_index(drop=True)
            prep_filtered = prep_filtered.reset_index(drop=True)
            df = pd.concat([raw, prep_filtered], axis=1)

        # Clean column names
        if 'cluster' not in df.columns:
            cluster_cols = [col for col in df.columns if 'cluster' in col.lower()]
            if cluster_cols:
                df['cluster'] = df[cluster_cols[0]]

        if 'outlier_flag' not in df.columns:
            outlier_cols = [col for col in df.columns if 'outlier' in col.lower()]
            if outlier_cols:
                df['outlier_flag'] = df[outlier_cols[0]].astype(str)
        else:
            df['outlier_flag'] = df['outlier_flag'].astype(str)

        # Clean cluster names
        df['cluster_clean'] = df['cluster'].astype(str).str.replace('cluster_', '', regex=False)
        df['cluster_clean'] = df['cluster_clean'].str.replace('Cluster ', '', regex=False)
        
        # Create boolean outlier column
        df['is_outlier'] = df['outlier_flag'].astype(str).str.lower().isin(['outlier', '1', 'true', 'yes'])
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def load_backtest_results():
    """Load backtesting results for comparison"""
    try:
        results = {}
        
        # Load results with outlier filter
        try:
            filtered_df = pd.read_csv('backtesting/optimization_results_filtered.csv')
            results['filtered'] = filtered_df.iloc[0].to_dict()
        except FileNotFoundError:
            results['filtered'] = None
        
        # Load results without outlier filter  
        try:
            original_df = pd.read_csv('backtesting/optimization_results_original.csv')
            results['original'] = original_df.iloc[0].to_dict()
        except FileNotFoundError:
            results['original'] = None
        
        return results
        
    except Exception as e:
        st.error(f"Error loading backtesting results: {str(e)}")
        return {'filtered': None, 'original': None}

@st.cache_data
def load_cluster_results():
    """Load cluster optimization results"""
    try:
        import json
        
        # Load results with outlier filter
        try:
            with open('backtesting/cluster_optimization_results_filtered.json', 'r') as f:
                cluster_results = json.load(f)
            return cluster_results
        except FileNotFoundError:
            return None
        
    except Exception as e:
        st.error(f"Error loading cluster results: {str(e)}")
        return None

def get_cluster_colors():
    """Define specific colors for clusters"""
    return {
        '0': '#1f77b4',  # Blue
        '1': '#ffbb33',  # Yellow/Orange
        '2': '#2ca02c',  # Green
        '3': '#d62728',  # Red
        '4': '#9467bd',  # Purple
        '5': '#8c564b',  # Brown
        '6': '#e377c2',  # Pink
        '7': '#7f7f7f',  # Gray
        '8': '#bcbd22',  # Olive
        '9': '#17becf'   # Cyan
    }

def create_candlestick_chart(df, selected_clusters, show_outliers_only, outlier_display):
    """Create candlestick chart with Plotly"""
    
    # Filter by selected clusters
    if selected_clusters:
        df_filtered = df[df['cluster_clean'].isin([str(c) for c in selected_clusters])].copy()
    else:
        df_filtered = df.copy()
    
    # Filter by outliers if enabled
    if show_outliers_only:
        df_filtered = df_filtered[df_filtered['is_outlier'] == True]
    
    if len(df_filtered) == 0:
        st.warning("No data to display with selected filters.")
        return None
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('EUR/USD Price', 'Cluster Information'),
        row_heights=[0.8, 0.2]
    )
    
    cluster_colors = get_cluster_colors()
    unique_clusters = sorted(df_filtered['cluster_clean'].unique())
    
    # Add candlestick per cluster
    for i, cluster in enumerate(unique_clusters):
        cluster_data = df_filtered[df_filtered['cluster_clean'] == cluster]
        
        color = cluster_colors.get(cluster, px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)])
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=cluster_data['time'],
                open=cluster_data['open'],
                high=cluster_data['high'],
                low=cluster_data['low'],
                close=cluster_data['close'],
                name=f'Cluster {cluster}',
                increasing_line_color=color,
                decreasing_line_color=color,
                increasing_fillcolor=color,
                decreasing_fillcolor=color,
                opacity=0.8
            ),
            row=1, col=1
        )
        
        # Scatter plot to show clusters in bottom panel
        fig.add_trace(
            go.Scatter(
                x=cluster_data['time'],
                y=[i] * len(cluster_data),
                mode='markers',
                marker=dict(color=color, size=8, symbol='circle'),
                name=f'Cluster {cluster} Timeline',
                showlegend=False,
                hovertemplate=f'<b>Cluster {cluster}</b><br>' +
                             'Time: %{x}<br>' +
                             '<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Add outlier markers if enabled
    if not show_outliers_only and any(df_filtered['is_outlier']):
        outliers = df_filtered[df_filtered['is_outlier'] == True]
        
        if outlier_display == "Markers":
            fig.add_trace(
                go.Scatter(
                    x=outliers['time'],
                    y=outliers['high'] * 1.001,  # Slightly above the high
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    name='Outliers',
                    hovertemplate='<b>OUTLIER</b><br>' +
                                 'Time: %{x}<br>' +
                                 'Cluster: %{customdata}<br>' +
                                 '<extra></extra>',
                    customdata=outliers['cluster_clean']
                ),
                row=1, col=1
            )
        elif outlier_display == "Vertical lines":
            for _, outlier in outliers.iterrows():
                fig.add_vline(
                    x=outlier['time'],
                    line=dict(color="red", width=2, dash="dash"),
                    opacity=0.7,
                    row=1, col=1
                )
    
    # Configure layout
    fig.update_layout(
        title=dict(
            text="📈 EUR/USD 15M Analysis - Trading with Clusters",
            x=0.5,
            font=dict(size=20, color='#2c3e50')
        ),
        xaxis_rangeslider_visible=False,
        height=700,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Configure axes
    fig.update_xaxes(
        title_text="Date and Time",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        row=2, col=1
    )
    
    fig.update_yaxes(
        title_text="Price",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Clusters",
        showgrid=False,
        tickmode='array',
        tickvals=list(range(len(unique_clusters))),
        ticktext=[f'Cluster {c}' for c in unique_clusters],
        row=2, col=1
    )
    
    return fig

def recreate_portfolio_from_results(backtest_result, df_data):
    """
    Recreate a vectorbt portfolio using backtesting results parameters
    """
    if not BACKTESTING_AVAILABLE or backtest_result is None:
        return None
    
    try:
        # Create model with optimized parameters
        ema_period = int(backtest_result['ema_period'])
        use_outlier_filter = bool(backtest_result['use_outlier_filter'])
        
        model = EMAModel(ema_period=ema_period)
        
        # Generate signals
        data_with_signals = model.generate_signals(df_data)
        
        # Shift signals to simulate execution on next candle
        shifted_signals = shift_signals(data_with_signals['signal'])
        
        # Convert signals to entries/exits for vectorbt
        entries = shifted_signals == 1
        exits = shifted_signals == -1
        
        # Force close open positions to avoid overlaps
        entries, exits = force_close_open_positions_numba(entries.values, exits.values)
        
        # Convert from numpy arrays back to pandas Series to maintain indices
        entries = pd.Series(entries, index=df_data.index)
        exits = pd.Series(exits, index=df_data.index)
        
        # Filter entries on outliers only if enabled
        if use_outlier_filter:
            entries = filter_outlier_entries(entries, df_data['outlier_flag'])
        
        # Execute backtesting with vectorbt
        portfolio = vbt.Portfolio.from_signals(
            df_data['close'],
            entries=entries,
            exits=exits,
            init_cash=1_000_000  # 1M for better visualization
        )
        
        return portfolio
        
    except Exception as e:
        st.error(f"Error recreating portfolio: {str(e)}")
        return None

def create_vectorbt_charts(portfolio, title_suffix=""):
    """
    Create vectorbt charts for orders and cumulative returns
    """
    if portfolio is None:
        return None, None
    
    try:
        # Orders chart (trades)
        orders_fig = go.Figure()
        
        # Add base price
        orders_fig.add_trace(go.Scatter(
            x=portfolio.wrapper.index,
            y=portfolio.close,
            mode='lines',
            name='Price',
            line=dict(color='black', width=1),
            opacity=0.7
        ))
        
        # Try to add orders if available
        try:
            if hasattr(portfolio, 'orders') and len(portfolio.orders.records_readable) > 0:
                orders_data = portfolio.orders.records_readable
                
                # Check if we have the 'Side' column (that we saw in debug)
                if 'Side' in orders_data.columns:
                    # Use 'Timestamp' instead of 'idx' for X axis
                    x_column = 'Timestamp' if 'Timestamp' in orders_data.columns else orders_data.index
                    
                    # Add buy orders
                    buy_orders = orders_data[orders_data['Side'] == 'Buy']
                    if not buy_orders.empty:
                        orders_fig.add_trace(go.Scatter(
                            x=buy_orders[x_column] if isinstance(x_column, str) else x_column[buy_orders.index],
                            y=buy_orders['Price'],
                            mode='markers',
                            name='Buys',
                            marker=dict(
                                symbol='triangle-up',
                                size=12,
                                color='green',
                                line=dict(width=2, color='darkgreen')
                            ),
                            hovertemplate='<b>Buy</b><br>' +
                                        'Date: %{x}<br>' +
                                        'Price: %{y:.5f}<br>' +
                                        '<extra></extra>'
                        ))
                    
                    # Add sell orders
                    sell_orders = orders_data[orders_data['Side'] == 'Sell']
                    if not sell_orders.empty:
                        orders_fig.add_trace(go.Scatter(
                            x=sell_orders[x_column] if isinstance(x_column, str) else x_column[sell_orders.index],
                            y=sell_orders['Price'],
                            mode='markers',
                            name='Sells',
                            marker=dict(
                                symbol='triangle-down',
                                size=12,
                                color='red',
                                line=dict(width=2, color='darkred')
                            ),
                            hovertemplate='<b>Sell</b><br>' +
                                        'Date: %{x}<br>' +
                                        'Price: %{y:.5f}<br>' +
                                        '<extra></extra>'
                        ))
                    
                    st.success(f"✅ Orders loaded: {len(buy_orders)} buys, {len(sell_orders)} sells")
                
                elif 'size' in orders_data.columns:
                    # Use size to determine direction
                    x_column = 'Timestamp' if 'Timestamp' in orders_data.columns else orders_data.index
                    
                    buy_orders = orders_data[orders_data['size'] > 0]
                    sell_orders = orders_data[orders_data['size'] < 0]
                    
                    if not buy_orders.empty:
                        orders_fig.add_trace(go.Scatter(
                            x=buy_orders[x_column] if isinstance(x_column, str) else x_column[buy_orders.index],
                            y=buy_orders['Price'],
                            mode='markers',
                            name='Buys',
                            marker=dict(symbol='triangle-up', size=12, color='green')
                        ))
                    
                    if not sell_orders.empty:
                        orders_fig.add_trace(go.Scatter(
                            x=sell_orders[x_column] if isinstance(x_column, str) else x_column[sell_orders.index],
                            y=sell_orders['Price'],
                            mode='markers',
                            name='Sells',
                            marker=dict(symbol='triangle-down', size=12, color='red')
                        ))
                
                else:
                    # Show all orders as generic points
                    x_column = 'Timestamp' if 'Timestamp' in orders_data.columns else orders_data.index
                    orders_fig.add_trace(go.Scatter(
                        x=orders_data[x_column] if isinstance(x_column, str) else x_column,
                        y=orders_data['Price'],
                        mode='markers',
                        name='Orders',
                        marker=dict(symbol='circle', size=10, color='blue')
                    ))
            else:
                st.info("No orders recorded in portfolio")
        except Exception as e:
            st.warning(f"Could not load orders: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        
        orders_fig.update_layout(
            title=f'📊 Trading Orders {title_suffix}',
            xaxis_title='Date',
            yaxis_title='Price',
            height=400,
            showlegend=True
        )

        return orders_fig, None

    except Exception as e:
        st.error(f"Error creating vectorbt charts: {str(e)}")
        return None, None

def create_individual_cluster_charts(cluster_portfolios):
    """
    Create individual charts for each cluster showing when each strategy acts
    """
    if not cluster_portfolios:
        return None
    
    cluster_colors = {
        '0': '#2ecc71',  # Green
        '1': '#3498db',  # Blue
        '2': '#e74c3c',  # Red
        '3': '#f39c12',  # Orange
        '4': '#9b59b6',  # Purple
        '5': '#1abc9c'   # Turquoise
    }
    
    charts_data = {}
    
    for cluster_id, cluster_info in cluster_portfolios.items():
        try:
            portfolio = cluster_info['portfolio']
            strategy = cluster_info['strategy']
            params = cluster_info['params']
            data = cluster_info['data']
            entries = cluster_info['entries']
            exits = cluster_info['exits']
            cluster_num = cluster_info['cluster_num']
            
            # Create chart for this cluster
            fig = go.Figure()
            
            # Add candlestick
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name=f'Price - Cluster {cluster_num}',
                increasing_line_color=cluster_colors.get(cluster_num, '#2ecc71'),
                decreasing_line_color=cluster_colors.get(cluster_num, '#2ecc71'),
                opacity=0.7
            ))
            
            # Add entry signals (buy) - only if few signals to improve performance
            entry_points = data[entries]
            if len(entry_points) > 0:
                # Limit number of signals shown to improve performance
                max_signals = 200
                if len(entry_points) > max_signals:
                    step = len(entry_points) // max_signals
                    entry_points = entry_points.iloc[::step]
                
                fig.add_trace(go.Scatter(
                    x=entry_points.index,
                    y=entry_points['low'] * 0.999,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=8,  # Reduced from 12 to 8
                        color='green',
                        line=dict(width=1, color='darkgreen')  # Reduced from 2 to 1
                    ),
                    name=f'Buy ({strategy})',
                    hovertemplate='<b>BUY</b><br>%{x}<br>%{y:.5f}<extra></extra>'
                ))
            
            # Add exit signals (sell)
            exit_points = data[exits]
            if len(exit_points) > 0:
                # Limit number of signals shown to improve performance
                if len(exit_points) > max_signals:
                    step = len(exit_points) // max_signals
                    exit_points = exit_points.iloc[::step]
                
                fig.add_trace(go.Scatter(
                    x=exit_points.index,
                    y=exit_points['high'] * 1.001,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=8,  # Reduced from 12 to 8
                        color='red',
                        line=dict(width=1, color='darkred')  # Reduced from 2 to 1
                    ),
                    name=f'Sell ({strategy})',
                    hovertemplate='<b>SELL</b><br>%{x}<br>%{y:.5f}<extra></extra>'
                ))
            
            # Configure layout
            params_text = ", ".join([f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" 
                                   for k, v in params.items()])
            
            fig.update_layout(
                title=f'🎯 Cluster {cluster_num} - Strategy {strategy}<br>' +
                      f'<sub>Parameters: {params_text}</sub>',
                xaxis_title='Date',
                yaxis_title='Price',
                height=450,
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Arial, sans-serif"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            )
            
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            )
            
            # Calculate metrics for this cluster
            total_return = portfolio.total_return()
            max_drawdown = abs(portfolio.max_drawdown())
            
            try:
                vbt.settings.array_wrapper['freq'] = '15min'
                sharpe = portfolio.sharpe_ratio()
            except:
                sharpe = 0.0
            
            # Count signals
            num_entries = entries.sum()
            num_exits = exits.sum()
            
            metrics = {
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe,
                'num_entries': num_entries,
                'num_exits': num_exits,
                'num_candles': len(data)
            }
            
            charts_data[cluster_id] = {
                'figure': fig,
                'metrics': metrics,
                'strategy': strategy,
                'params': params,
                'cluster_num': cluster_num
            }
            
        except Exception as e:
            st.warning(f"Error creating chart for {cluster_id}: {str(e)}")
            continue
    
    return charts_data

def recreate_cluster_portfolio(cluster_results, df_data):
    """
    Recreate a combined portfolio using cluster-optimized strategies
    """
    if not BACKTESTING_AVAILABLE or cluster_results is None:
        return None
    
    try:
        # Process cluster columns same as in load_data()
        if 'cluster_clean' not in df_data.columns:
            if 'cluster' in df_data.columns:
                df_data['cluster_clean'] = df_data['cluster'].astype(str).str.replace('cluster_', '', regex=False)
                df_data['cluster_clean'] = df_data['cluster_clean'].str.replace('Cluster ', '', regex=False)
            else:
                st.error("No 'cluster' column found in data")
                return None
        
        # Get weight configuration
        weight_config = cluster_results.get('weight_config', {})
        if not weight_config:
            st.error("No weight configuration found in results")
            return None
        
        # Create strategy dictionary
        strategy_classes = {
            'EMA': EMAStrategy,
            'RSI': RSIStrategy,
            'MACD': MACDStrategy,
            'Bollinger': BollingerStrategy,
            'Stochastic': StochasticStrategy
        }
        
        # Generate combined signals
        all_entries = pd.Series(False, index=df_data.index)
        all_exits = pd.Series(False, index=df_data.index)
        
        for cluster_id, config in weight_config.items():
            # Filter data by cluster
            cluster_num = cluster_id.replace('cluster_', '')
            cluster_mask = df_data['cluster_clean'] == cluster_num
            cluster_data = df_data[cluster_mask].copy()
            
            if len(cluster_data) == 0:
                continue
            
            # Get best strategy for this cluster
            best_strategy = config.get('best_strategy')
            best_params = config.get('best_params', {})
            
            if best_strategy in strategy_classes:
                # Create strategy instance with optimized parameters
                strategy = strategy_classes[best_strategy]()
                strategy.set_params(**best_params)
                
                # Generate signals for this cluster
                cluster_signals = strategy.generate_signals(cluster_data)
                
                # Map signals to complete DataFrame
                if 'signal' in cluster_signals.columns:
                    # Apply signals only where cluster corresponds
                    cluster_entries = (cluster_signals['signal'] == 1)
                    cluster_exits = (cluster_signals['signal'] == -1)
                    
                    # Map to complete index
                    all_entries.loc[cluster_data.index] |= cluster_entries
                    all_exits.loc[cluster_data.index] |= cluster_exits
        
        # Shift signals to simulate execution on next candle
        shifted_entries = shift_signals(all_entries.astype(int))
        shifted_exits = shift_signals(all_exits.astype(int))
        
        entries = shifted_entries == 1
        exits = shifted_exits == 1
        
        # Force close open positions to avoid overlaps
        entries_array, exits_array = force_close_open_positions_numba(entries.values, exits.values)
        
        # Convert back to pandas Series
        entries = pd.Series(entries_array, index=df_data.index)
        exits = pd.Series(exits_array, index=df_data.index)
        
        # Apply outlier filter if configured
        config_info = cluster_results.get('configuration', {})
        if config_info.get('use_outlier_filter', False):
            entries = filter_outlier_entries(entries, df_data['outlier_flag'])
        
        # Create portfolio with vectorbt
        portfolio = vbt.Portfolio.from_signals(
            df_data['close'],
            entries=entries,
            exits=exits,
            init_cash=1_000_000
        )
        
        return portfolio
        
    except Exception as e:
        st.error(f"Error recreating cluster portfolio: {str(e)}")
        return None

def create_individual_cluster_portfolios(cluster_results, df_data):
    """
    Crea portfolios individuales para cada cluster con sus estrategias específicas
    """
    if not BACKTESTING_AVAILABLE or cluster_results is None:
        return {}
    
    try:
        # Procesar columnas de cluster igual que en recreate_cluster_portfolio
        if 'cluster_clean' not in df_data.columns:
            if 'cluster' in df_data.columns:
                df_data['cluster_clean'] = df_data['cluster'].astype(str).str.replace('cluster_', '', regex=False)
                df_data['cluster_clean'] = df_data['cluster_clean'].str.replace('Cluster ', '', regex=False)
            else:
                return {}
        
        # Obtener configuración de weight
        weight_config = cluster_results.get('weight_config', {})
        if not weight_config:
            return {}
        
        # Crear diccionario de estrategias
        strategy_classes = {
            'EMA': EMAStrategy,
            'RSI': RSIStrategy,
            'MACD': MACDStrategy,
            'Bollinger': BollingerStrategy,
            'Stochastic': StochasticStrategy
        }
        
        cluster_portfolios = {}
        
        for cluster_id, config in weight_config.items():
            try:
                # Filtrar datos por cluster
                cluster_num = cluster_id.replace('cluster_', '')
                cluster_mask = df_data['cluster_clean'] == cluster_num
                cluster_data = df_data[cluster_mask].copy()
                
                if len(cluster_data) == 0:
                    continue
                
                # Obtener la mejor estrategia para este cluster
                best_strategy = config.get('best_strategy')
                best_params = config.get('best_params', {})
                
                if best_strategy in strategy_classes:
                    # Crear instancia de la estrategia con parámetros optimizados
                    strategy = strategy_classes[best_strategy]()
                    strategy.set_params(**best_params)
                    
                    # Generar señales para este cluster
                    cluster_signals = strategy.generate_signals(cluster_data)
                    
                    if 'signal' in cluster_signals.columns:
                        # Convertir señales a entradas/salidas
                        entries = (cluster_signals['signal'] == 1)
                        exits = (cluster_signals['signal'] == -1)
                        
                        # Desplazar señales para simular ejecución en siguiente vela
                        shifted_entries = shift_signals(entries.astype(int))
                        shifted_exits = shift_signals(exits.astype(int))
                        
                        entries = shifted_entries == 1
                        exits = shifted_exits == 1
                        
                        # Forzar cierre de posiciones abiertas
                        entries_array, exits_array = force_close_open_positions_numba(entries.values, exits.values)
                        
                        # Convertir de vuelta a pandas Series
                        entries = pd.Series(entries_array, index=cluster_data.index)
                        exits = pd.Series(exits_array, index=cluster_data.index)
                        
                        # Aplicar filtro de outliers si está configurado
                        config_info = cluster_results.get('configuration', {})
                        if config_info.get('use_outlier_filter', False):
                            entries = filter_outlier_entries(entries, cluster_data['outlier_flag'])
                        
                        # Crear portfolio individual
                        portfolio = vbt.Portfolio.from_signals(
                            cluster_data['close'],
                            entries=entries,
                            exits=exits,
                            init_cash=1_000_000
                        )
                        
                        cluster_portfolios[cluster_id] = {
                            'portfolio': portfolio,
                            'strategy': best_strategy,
                            'params': best_params,
                            'data': cluster_data,
                            'entries': entries,
                            'exits': exits,
                            'cluster_num': cluster_num
                        }
                        
            except Exception as e:
                st.warning(f"Error procesando cluster {cluster_id}: {str(e)}")
                continue
        
        return cluster_portfolios
        
    except Exception as e:
        st.error(f"Error creando portfolios individuales: {str(e)}")
        return {}

def show_statistics(df, selected_clusters, show_outliers_only):
    """Show data statistics"""
    
    # Filter data according to selection
    if selected_clusters:
        df_filtered = df[df['cluster_clean'].isin([str(c) for c in selected_clusters])].copy()
    else:
        df_filtered = df.copy()
    
    if show_outliers_only:
        df_filtered = df_filtered[df_filtered['is_outlier'] == True]
    
    if len(df_filtered) == 0:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Total Candles</h4>
            <h2>{:,}</h2>
        </div>
        """.format(len(df_filtered)), unsafe_allow_html=True)
    
    with col2:
        outliers_count = df_filtered['is_outlier'].sum()
        outliers_pct = (outliers_count / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
        st.markdown("""
        <div class="metric-card">
            <h4>🚨 Outliers</h4>
            <h2>{} ({:.1f}%)</h2>
        </div>
        """.format(outliers_count, outliers_pct), unsafe_allow_html=True)
    
    with col3:
        clusters_count = len(df_filtered['cluster_clean'].unique())
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Active Clusters</h4>
            <h2>{}</h2>
        </div>
        """.format(clusters_count), unsafe_allow_html=True)
    
    with col4:
        price_range = df_filtered['high'].max() - df_filtered['low'].min()
        st.markdown("""
        <div class="metric-card">
            <h4>📈 Price Range</h4>
            <h2>{:.5f}</h2>
        </div>
        """.format(price_range), unsafe_allow_html=True)

def show_backtest_comparison(backtest_results):
    """Show backtesting results comparison"""
    
    st.markdown("## 🎯 Trading Models Comparison")
    st.markdown("### EMA Model with and without outlier filter")
    
    filtered_results = backtest_results.get('filtered')
    original_results = backtest_results.get('original')
    
    if not filtered_results and not original_results:
        st.warning("No backtesting results found. Run the train.py script first.")
        return
    
    # Create columns for comparison
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("### 🔒 With Outlier Filter")
        if filtered_results:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2ecc71, #27ae60); 
                       color: white; padding: 1.5rem; border-radius: 1rem; margin: 1rem 0;">
                <h4 style="margin: 0 0 1rem 0; color: white;">📊 Main Metrics</h4>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>EMA Period:</strong> {filtered_results['ema_period']}</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Total Return:</strong> {filtered_results['total_return']*100:.2f}%</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Max Drawdown:</strong> {filtered_results['max_drawdown']*100:.2f}%</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Sharpe Ratio:</strong> {filtered_results['sharpe_ratio']:.3f}</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Total Trades:</strong> {filtered_results['total_trades']:,}</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Win Rate:</strong> {filtered_results['win_rate']*100:.1f}%</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Objective Value:</strong> {filtered_results['objective_value']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Filtered results not available")
    
    with col2:
        st.markdown("### 🔓 Without Outlier Filter")
        if original_results:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #3498db, #2980b9); 
                       color: white; padding: 1.5rem; border-radius: 1rem; margin: 1rem 0;">
                <h4 style="margin: 0 0 1rem 0; color: white;">📊 Main Metrics</h4>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>EMA Period:</strong> {original_results['ema_period']}</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Total Return:</strong> {original_results['total_return']*100:.2f}%</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Max Drawdown:</strong> {original_results['max_drawdown']*100:.2f}%</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Sharpe Ratio:</strong> {original_results['sharpe_ratio']:.3f}</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Total Trades:</strong> {original_results['total_trades']:,}</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Win Rate:</strong> {original_results['win_rate']*100:.1f}%</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Objective Value:</strong> {original_results['objective_value']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Unfiltered results not available")
    
    with col3:
        st.markdown("### 📈 Comparative Analysis")
        if filtered_results and original_results:
            # Calculate differences
            return_diff = (filtered_results['total_return'] - original_results['total_return']) * 100
            drawdown_diff = (filtered_results['max_drawdown'] - original_results['max_drawdown']) * 100
            sharpe_diff = filtered_results['sharpe_ratio'] - original_results['sharpe_ratio']
            trades_diff = filtered_results['total_trades'] - original_results['total_trades']
            winrate_diff = (filtered_results['win_rate'] - original_results['win_rate']) * 100
            objective_diff = filtered_results['objective_value'] - original_results['objective_value']
            
            # Determine colors for differences
            return_color = "green" if return_diff > 0 else "red"
            drawdown_color = "green" if drawdown_diff < 0 else "red"  # Lower drawdown is better
            sharpe_color = "green" if sharpe_diff > 0 else "red"
            objective_color = "green" if objective_diff > 0 else "red"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f39c12, #e67e22); 
                       color: white; padding: 1.5rem; border-radius: 1rem; margin: 1rem 0;">
                <h4 style="margin: 0 0 1rem 0; color: white;">🔍 Differences (Filter vs Original)</h4>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Return:</strong> <span style="color: {return_color};">{return_diff:+.2f}%</span></p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Max Drawdown:</strong> <span style="color: {drawdown_color};">{drawdown_diff:+.2f}%</span></p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Sharpe Ratio:</strong> <span style="color: {sharpe_color};">{sharpe_diff:+.3f}</span></p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Total Trades:</strong> {trades_diff:+,}</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Win Rate:</strong> {winrate_diff:+.1f}%</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Objective Value:</strong> <span style="color: {objective_color};">{objective_diff:+.4f}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Conclusion
            st.markdown("### 🎯 Conclusion")
            if objective_diff > 0:
                st.success(f"✅ **The model with outlier filter is {objective_diff:.2%} better** according to the objective metric (Return - Max Drawdown)")
            else:
                st.warning(f"⚠️ **The model without outlier filter is {abs(objective_diff):.2%} better** according to the objective metric (Return - Max Drawdown)")
        else:
            st.info("You need both results to compare")
    
    # Comparative chart
    if filtered_results and original_results:
        st.markdown("### 📊 Comparative Visualization")
        
        # Create comparative bar chart
        metrics = ['Total Return (%)', 'Max Drawdown (%)', 'Sharpe Ratio', 'Win Rate (%)']
        filtered_values = [
            filtered_results['total_return'] * 100,
            filtered_results['max_drawdown'] * 100,
            filtered_results['sharpe_ratio'],
            filtered_results['win_rate'] * 100
        ]
        original_values = [
            original_results['total_return'] * 100,
            original_results['max_drawdown'] * 100,
            original_results['sharpe_ratio'],
            original_results['win_rate'] * 100
        ]
        
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Bar(
            name='With Outlier Filter',
            x=metrics,
            y=filtered_values,
            marker_color='#2ecc71',
            text=[f'{v:.2f}' for v in filtered_values],
            textposition='auto'
        ))
        
        fig_comparison.add_trace(go.Bar(
            name='Without Outlier Filter',
            x=metrics,
            y=original_values,
            marker_color='#3498db',
            text=[f'{v:.2f}' for v in original_values],
            textposition='auto'
        ))
        
        fig_comparison.update_layout(
            title='📊 Trading Metrics Comparison',
            xaxis_title='Metrics',
            yaxis_title='Values',
            barmode='group',
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Vectorbt charts
        if BACKTESTING_AVAILABLE and filtered_results and original_results:
            st.markdown("### 📈 Detailed Trading Analysis")
            
            # Load data to recreate portfolios
            try:
                data_path = Path(__file__).parent / "data_preprocessed" / "enhanced_eurusd_dataset.csv"
                if data_path.exists():
                    trading_data = load_backtest_data(str(data_path))
                    
                    # Use only a sample for visualization (last 10k records)
                    sample_size = min(10000, len(trading_data))
                    trading_data = trading_data.iloc[-sample_size:].copy()
                    
                    st.info(f"📊 Generating charts with {len(trading_data):,} records (latest data)")
                    
                    # Recreate portfolios
                    with st.spinner("Recreating portfolios..."):
                        portfolio_filtered = recreate_portfolio_from_results(filtered_results, trading_data)
                        portfolio_original = recreate_portfolio_from_results(original_results, trading_data)
                    
                    # Crear gráficas
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🔒 Modelo con Filtro de Outliers")
                        if portfolio_filtered is not None:
                            orders_fig_filtered, cumulative_fig_filtered = create_vectorbt_charts(
                                portfolio_filtered, "(Con Filtro)"
                            )
                            
                            if orders_fig_filtered:
                                st.plotly_chart(orders_fig_filtered, use_container_width=True)
                        else:
                            st.error("No se pudo recrear el portfolio con filtro")
                    
                    with col2:
                        st.markdown("#### 🔓 Modelo sin Filtro de Outliers")
                        if portfolio_original is not None:
                            orders_fig_original, cumulative_fig_original = create_vectorbt_charts(
                                portfolio_original, "(Sin Filtro)"
                            )
                            
                            if orders_fig_original:
                                st.plotly_chart(orders_fig_original, use_container_width=True)
                        else:
                            st.error("No se pudo recrear el portfolio sin filtro")
                    
                    # Cumulative returns comparison in a single chart
                    if portfolio_filtered is not None and portfolio_original is not None:
                        st.markdown("#### 📊 Direct Returns Comparison")
                        
                        comparison_returns_fig = go.Figure()
                        
                        # Portfolio with filter
                        comparison_returns_fig.add_trace(go.Scatter(
                            x=portfolio_filtered.wrapper.index,
                            y=portfolio_filtered.value(),
                            mode='lines',
                            name='With Outlier Filter',
                            line=dict(color='#2ecc71', width=3)
                        ))
                        
                        # Portfolio without filter
                        comparison_returns_fig.add_trace(go.Scatter(
                            x=portfolio_original.wrapper.index,
                            y=portfolio_original.value(),
                            mode='lines',
                            name='Without Outlier Filter',
                            line=dict(color='#3498db', width=3)
                        ))
                        
                        # Initial capital baseline
                        initial_cash = portfolio_filtered.init_cash
                        comparison_returns_fig.add_hline(
                            y=initial_cash,
                            line_dash="dash",
                            line_color="gray",
                            annotation_text=f"Initial Capital: ${initial_cash:,.0f}"
                        )
                        
                        comparison_returns_fig.update_layout(
                            title='💰 Cumulative Returns Comparison',
                            xaxis_title='Date',
                            yaxis_title='Portfolio Value ($)',
                            yaxis=dict(tickformat='$,.0f'),  # Fixed money format
                            height=500,
                            showlegend=True,
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )
                        
                        st.plotly_chart(comparison_returns_fig, use_container_width=True)
                
                else:
                    st.warning("Preprocessed data file not found to generate trading charts")
                    
            except Exception as e:
                st.error(f"Error generating trading charts: {str(e)}")
        
        elif not BACKTESTING_AVAILABLE:
            st.info("Detailed trading charts require backtesting modules.")

def show_cluster_strategy_results(cluster_results):
    """Show cluster optimization results comparing with single strategy"""
    
    st.markdown("## 🎯 Trading Models Comparison")
    st.markdown("### Single strategy (EMA) vs Cluster-adapted strategies")
    
    if not cluster_results:
        st.warning("No cluster optimization results found. Run the cluster_train.py script first.")
        return
    
    # Also load backtesting results for comparison
    backtest_results = load_backtest_results()
    
    # Get combined cluster model metrics
    combined_perf = cluster_results.get('combined_performance', {})
    
    # Get EMA model metrics with filter (best of the two previous)
    ema_results = backtest_results.get('filtered') or backtest_results.get('original')
    
    if not combined_perf and not ema_results:
        st.warning("Not enough data found for comparison.")
        return
    
    # Create columns for side-by-side comparison
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("### 📊 Single Strategy (EMA)")
        if ema_results:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #3498db, #2980b9); 
                       color: white; padding: 1.5rem; border-radius: 1rem; margin: 1rem 0;">
                <h4 style="margin: 0 0 1rem 0; color: white;">📊 Main Metrics</h4>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>EMA Period:</strong> {ema_results['ema_period']}</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Total Return:</strong> {ema_results['total_return']*100:.2f}%</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Max Drawdown:</strong> {ema_results['max_drawdown']*100:.2f}%</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Sharpe Ratio:</strong> {ema_results['sharpe_ratio']:.3f}</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Total Trades:</strong> {ema_results['total_trades']:,}</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Win Rate:</strong> {ema_results['win_rate']*100:.1f}%</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Objective Value:</strong> {ema_results['objective_value']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Single strategy results not available")
    
    with col2:
        st.markdown("### 🧠 Cluster-Adapted Strategies")
        if combined_perf:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2ecc71, #27ae60); 
                       color: white; padding: 1.5rem; border-radius: 1rem; margin: 1rem 0;">
                <h4 style="margin: 0 0 1rem 0; color: white;">📊 Main Metrics</h4>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Multiple Strategies</strong></p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Total Return:</strong> {combined_perf.get('total_return', 0)*100:.2f}%</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Max Drawdown:</strong> {combined_perf.get('max_drawdown', 0)*100:.2f}%</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Sharpe Ratio:</strong> {combined_perf.get('sharpe_ratio', 0):.3f}</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Total Trades:</strong> {combined_perf.get('total_trades', 0):,}</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Win Rate:</strong> {combined_perf.get('win_rate', 0)*100:.1f}%</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Objective Value:</strong> {combined_perf.get('objective_value', 0):.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Adapted strategies results not available")
    
    with col3:
        st.markdown("### 📈 Comparative Analysis")
        if combined_perf and ema_results:
            # Calculate differences
            return_diff = (combined_perf.get('total_return', 0) - ema_results['total_return']) * 100
            drawdown_diff = (combined_perf.get('max_drawdown', 0) - ema_results['max_drawdown']) * 100
            sharpe_diff = combined_perf.get('sharpe_ratio', 0) - ema_results['sharpe_ratio']
            trades_diff = combined_perf.get('total_trades', 0) - ema_results['total_trades']
            winrate_diff = (combined_perf.get('win_rate', 0) - ema_results['win_rate']) * 100
            objective_diff = combined_perf.get('objective_value', 0) - ema_results['objective_value']
            
            # Determine colors for differences
            return_color = "green" if return_diff > 0 else "red"
            drawdown_color = "green" if drawdown_diff < 0 else "red"  # Lower drawdown is better
            sharpe_color = "green" if sharpe_diff > 0 else "red"
            objective_color = "green" if objective_diff > 0 else "red"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f39c12, #e67e22); 
                       color: white; padding: 1.5rem; border-radius: 1rem; margin: 1rem 0;">
                <h4 style="margin: 0 0 1rem 0; color: white;">🔍 Differences (Clusters vs EMA)</h4>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Return:</strong> <span style="color: {return_color};">{return_diff:+.2f}%</span></p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Max Drawdown:</strong> <span style="color: {drawdown_color};">{drawdown_diff:+.2f}%</span></p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Sharpe Ratio:</strong> <span style="color: {sharpe_color};">{sharpe_diff:+.3f}</span></p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Total Trades:</strong> {trades_diff:+,}</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Win Rate:</strong> {winrate_diff:+.1f}%</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Objective Value:</strong> <span style="color: {objective_color};">{objective_diff:+.4f}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Conclusion
            st.markdown("### 🎯 Conclusion")
            if objective_diff > 0:
                st.success(f"✅ **Adapted strategies are {objective_diff:.2%} better** according to the objective metric")
            else:
                st.warning(f"⚠️ **Single EMA strategy is {abs(objective_diff):.2%} better** according to the objective metric")
        else:
            st.info("You need both results to compare")
    
    # Comparative bar chart
    if combined_perf and ema_results:
        st.markdown("### 📊 Comparative Visualization")
        
        metrics = ['Total Return (%)', 'Max Drawdown (%)', 'Sharpe Ratio', 'Win Rate (%)']
        ema_values = [
            ema_results['total_return'] * 100,
            ema_results['max_drawdown'] * 100,
            ema_results['sharpe_ratio'],
            ema_results['win_rate'] * 100
        ]
        cluster_values = [
            combined_perf.get('total_return', 0) * 100,
            combined_perf.get('max_drawdown', 0) * 100,
            combined_perf.get('sharpe_ratio', 0),
            combined_perf.get('win_rate', 0) * 100
        ]
        
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Bar(
            name='Single Strategy (EMA)',
            x=metrics,
            y=ema_values,
            marker_color='#3498db',
            text=[f'{v:.2f}' for v in ema_values],
            textposition='auto'
        ))
        
        fig_comparison.add_trace(go.Bar(
            name='Cluster-Adapted Strategies',
            x=metrics,
            y=cluster_values,
            marker_color='#2ecc71',
            text=[f'{v:.2f}' for v in cluster_values],
            textposition='auto'
        ))
        
        fig_comparison.update_layout(
            title='📊 Trading Metrics Comparison',
            xaxis_title='Metrics',
            yaxis_title='Values',
            barmode='group',
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Strategy Analysis during Training
        st.markdown("### 🔬 Strategy Analysis during Training")
        st.markdown("#### Performance comparison of all strategies tested per cluster")
        
        # Get cluster optimization results
        optimization_results = cluster_results.get('optimization_results', {})
        
        if optimization_results:
            # Create strategy performance chart per cluster
            fig_training = go.Figure()
            
            strategies = ['EMA', 'RSI', 'MACD', 'Bollinger', 'Stochastic']
            colors_strategies = {
                'EMA': '#FF6B35',
                'RSI': '#004E89', 
                'MACD': '#1A936F',
                'Bollinger': '#FF8C42',
                'Stochastic': '#7209B7'
            }
            
            # Add bars per strategy
            for strategy in strategies:
                cluster_values = []
                cluster_names = []
                
                for cluster_id, cluster_data in optimization_results.items():
                    cluster_num = cluster_id.replace('cluster_', '')
                    all_results = cluster_data.get('all_results', {})
                    
                    if strategy in all_results:
                        value = all_results[strategy].get('best_value', 0)
                        cluster_values.append(value)
                        cluster_names.append(f'Cluster {cluster_num}')
                    else:
                        cluster_values.append(0)
                        cluster_names.append(f'Cluster {cluster_num}')
                
                fig_training.add_trace(go.Bar(
                    name=strategy,
                    x=cluster_names,
                    y=cluster_values,
                    marker_color=colors_strategies.get(strategy, '#666666'),
                    text=[f'{v:.3f}' if v > 0 else 'N/A' for v in cluster_values],
                    textposition='auto',
                    opacity=0.8,
                    hovertemplate=f'<b>{strategy}</b><br>' +
                                 'Cluster: %{x}<br>' +
                                 'Objective Value: %{y:.4f}<br>' +
                                 '<extra></extra>'
                ))
            
            fig_training.update_layout(
                title='🔬 Strategy Performance during Training per Cluster',
                xaxis_title='Clusters',
                yaxis_title='Objective Value (Return - Max Drawdown)',
                barmode='group',
                height=600,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Arial, sans-serif"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            )
            
            # Add horizontal line at y=0 for reference
            fig_training.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray",
                opacity=0.5,
                annotation_text="Neutral value (0)"
            )
            
            st.plotly_chart(fig_training, use_container_width=True)
            
        # Detailed Trading Analysis (as in backtesting)
        if BACKTESTING_AVAILABLE:
            st.markdown("### 📈 Detailed Trading Analysis")
            
            try:
                data_path = Path(__file__).parent / "data_preprocessed" / "enhanced_eurusd_dataset.csv"
                if data_path.exists():
                    trading_data = load_backtest_data(str(data_path))
                    
                    # Use sample to improve performance
                    sample_size = min(10000, len(trading_data))
                    trading_data = trading_data.iloc[-sample_size:].copy()
                    
                    st.info(f"📊 Generating charts with {len(trading_data):,} records (latest data)")
                    
                    # Recreate portfolios
                    with st.spinner("Recreating portfolios for comparison..."):
                        # EMA Portfolio
                        portfolio_ema = recreate_portfolio_from_results(ema_results, trading_data)
                        # Clusters Portfolio
                        portfolio_clusters = recreate_cluster_portfolio(cluster_results, trading_data)
                    
                    # Create side-by-side charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if portfolio_ema is not None:
                            orders_fig_ema, returns_fig_ema = create_vectorbt_charts(portfolio_ema, "- Single EMA Strategy")
                            if orders_fig_ema is not None:
                                st.plotly_chart(orders_fig_ema, use_container_width=True)
                            if returns_fig_ema is not None:
                                st.plotly_chart(returns_fig_ema, use_container_width=True)
                        else:
                            st.warning("Could not generate EMA portfolio")
                    
                    with col2:
                        if portfolio_clusters is not None:
                            orders_fig_clusters, returns_fig_clusters = create_vectorbt_charts(portfolio_clusters, "- Cluster Strategies")
                            if orders_fig_clusters is not None:
                                st.plotly_chart(orders_fig_clusters, use_container_width=True)
                        else:
                            st.warning("Could not generate cluster portfolio")
                    
                    # Cumulative returns comparison
                    if portfolio_ema is not None and portfolio_clusters is not None:
                        st.markdown("### 💰 Cumulative Returns Comparison")
                        
                        fig_cumulative = go.Figure()
                        
                        # EMA Portfolio
                        ema_value = portfolio_ema.value()
                        fig_cumulative.add_trace(go.Scatter(
                            x=portfolio_ema.wrapper.index,
                            y=ema_value,
                            mode='lines',
                            name='Single Strategy (EMA)',
                            line=dict(width=3, color='#3498db')
                        ))
                        
                        # Clusters Portfolio
                        clusters_value = portfolio_clusters.value()
                        fig_cumulative.add_trace(go.Scatter(
                            x=portfolio_clusters.wrapper.index,
                            y=clusters_value,
                            mode='lines',
                            name='Adapted Strategies',
                            line=dict(width=3, color='#2ecc71')
                        ))
                        
                        # Baseline
                        initial_cash = portfolio_ema.init_cash
                        fig_cumulative.add_hline(
                            y=initial_cash,
                            line_dash="dash",
                            line_color="gray",
                            annotation_text=f"Initial Capital: ${initial_cash:,.0f}"
                        )
                        
                        fig_cumulative.update_layout(
                            title='💰 Cumulative Returns Comparison',
                            xaxis_title='Date',
                            yaxis_title='Portfolio Value ($)',
                            yaxis=dict(tickformat='$,.0f'),
                            height=500,
                            showlegend=True,
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )
                        
                        st.plotly_chart(fig_cumulative, use_container_width=True)
                
                else:
                    st.warning("Preprocessed data file not found to generate trading charts")
                    
            except Exception as e:
                st.error(f"Error generating trading charts: {str(e)}")
        
        elif not BACKTESTING_AVAILABLE:
            st.info("Detailed trading charts require backtesting modules.")
    
    # More sections can be added here as needed

def main():
    # Main title
    st.title("🚀 EUR/USD Trading Analysis Dashboard")
    st.markdown("### Advanced analysis with clustering and outlier detection")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
        backtest_results = load_backtest_results()
        cluster_results = load_cluster_results()
    
    if df is None:
        st.stop()
    
    # Tab navigation
    tab1, tab2, tab3 = st.tabs(["📈 Data Analysis", "🎯 Backtesting Results", "🧠 Cluster Strategies"])
    
    with tab1:
        # Sidebar for controls
        st.sidebar.header("⚙️ Configuration")
        
        # General information in sidebar
        st.sidebar.markdown(f"""
        **📋 Data Summary:**
        - **Total candles:** {len(df):,}
        - **Period:** {df['time'].min().strftime('%Y-%m-%d')} → {df['time'].max().strftime('%Y-%m-%d')}
        - **Available clusters:** {', '.join(sorted(df['cluster_clean'].unique()))}
        - **Total outliers:** {df['is_outlier'].sum():,} ({df['is_outlier'].sum()/len(df)*100:.1f}%)
        """)
        
        st.sidebar.markdown("---")
        
        # Cluster control
        available_clusters = sorted(df['cluster_clean'].unique())
        selected_clusters = st.sidebar.multiselect(
            "🎯 Select Clusters:",
            options=available_clusters,
            default=available_clusters,
            help="Select which clusters to show in the chart"
        )
        
        # Outliers control
        show_outliers_only = st.sidebar.checkbox(
            "🚨 Show Only Outliers",
            value=False,
            help="When activated, only shows candles marked as outliers"
        )
        
        # Outlier style (only if not in "outliers only" mode)
        if not show_outliers_only:
            outlier_display = st.sidebar.selectbox(
                "🎨 Outlier Style:",
                options=["Markers", "Vertical lines", "Don't show"],
                index=0,
                help="How to display outliers in the chart"
            )
        else:
            outlier_display = "Don't show"
        
        # Data range control
        st.sidebar.markdown("---")
        st.sidebar.subheader("📅 Data Range")
        
        # Slider to select data range
        date_range = st.sidebar.slider(
            "Select range:",
            min_value=0,
            max_value=len(df)-1,
            value=(0, min(1000, len(df)-1)),
            help="Slide to select the range of candles to display"
        )
        
        # Filter by date range
        df_range = df.iloc[date_range[0]:date_range[1]+1].copy()
        
        # Show statistics
        st.markdown("## 📊 Statistics")
        show_statistics(df_range, selected_clusters, show_outliers_only)
        
        # Cluster information
        if selected_clusters:
            st.markdown("## 🎯 Selected Clusters Information")
            
            cluster_colors = get_cluster_colors()
            cols = st.columns(min(len(selected_clusters), 4))
            
            for i, cluster in enumerate(selected_clusters):
                cluster_data = df_range[df_range['cluster_clean'] == str(cluster)]
                color = cluster_colors.get(str(cluster), '#666666')
                
                with cols[i % 4]:
                    outliers_in_cluster = cluster_data['is_outlier'].sum()
                    outliers_pct = (outliers_in_cluster / len(cluster_data) * 100) if len(cluster_data) > 0 else 0
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {color}20, {color}40); 
                               border-left: 4px solid {color}; 
                               padding: 1rem; 
                               border-radius: 0.5rem; 
                               margin: 0.5rem 0;">
                        <h4 style="color: {color}; margin: 0;">Cluster {cluster}</h4>
                        <p style="margin: 0.5rem 0 0 0;">
                            <strong>Candles:</strong> {len(cluster_data):,}<br>
                            <strong>Outliers:</strong> {outliers_in_cluster} ({outliers_pct:.1f}%)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Create and show chart
        st.markdown("## 📈 Candlestick Chart")
        
        fig = create_candlestick_chart(df_range, selected_clusters, show_outliers_only, outlier_display)
        
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional information
            with st.expander("ℹ️ Additional Information"):
                st.markdown("""
                **How to use this dashboard:**
                
                1. **Clusters:** Use the multiple selector in the sidebar to choose which clusters to display
                2. **Outliers:** Activate "Show Only Outliers" to see only anomalous candles
                3. **Styles:** Change how outliers are displayed (markers, lines, or don't show)
                4. **Range:** Adjust the slider to navigate through different time periods
                
                **Cluster Colors:**
                - 🔵 Cluster 0: Blue
                - 🟡 Cluster 1: Yellow/Orange
                - 🟢 Cluster 2+: Other colors
                
                **Markers:**
                - 🔺 Red triangles: Detected outliers
                - 📊 Bottom panel: Cluster timeline
                """)
    
    with tab2:
        # Show backtesting results
        show_backtest_comparison(backtest_results)
    
    with tab3:
        # Show cluster strategy results
        show_cluster_strategy_results(cluster_results)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        💡 Dashboard created with Streamlit and Plotly | 
        📈 EUR/USD 15M Analysis with Machine Learning
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        get_script_run_ctx = lambda: None

    ctx = get_script_run_ctx()

    if ctx is None:
        print("This application must be run with 'streamlit run app.py'.")
    else:
        main()
