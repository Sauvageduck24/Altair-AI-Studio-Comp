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

# Importaciones para backtesting
try:
    import vectorbt as vbt
    from pathlib import Path
    import sys
    
    # Agregar el directorio backtesting al path para importar módulos locales
    sys.path.append(str(Path(__file__).parent / "backtesting"))
    from model import EMAModel
    from utils import load_data as load_backtest_data, shift_signals, force_close_open_positions_numba, filter_outlier_entries
    BACKTESTING_AVAILABLE = True
except ImportError as e:
    BACKTESTING_AVAILABLE = False
    st.sidebar.warning("⚠️ Módulos de backtesting no disponibles. Algunas funciones estarán limitadas.")

# Configuración de la página
st.set_page_config(
    page_title="EUR/USD Trading Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar el estilo
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
    """Carga y prepara los datos con cache para optimizar performance"""
    try:
        # Cargar datos raw OHLC
        raw = pd.read_csv('data_raw/EURUSD_15M.csv')
        if 'time' not in raw.columns:
            st.error("El CSV de velas debe tener una columna 'time'.")
            return None
        
        raw['time'] = pd.to_datetime(raw['time'])
        for c in ['open','high','low','close']:
            raw[c] = pd.to_numeric(raw[c], errors='coerce')

        # Cargar datos preprocesados
        prep = pd.read_csv('data_preprocessed/results.csv', sep=';')
        
        # Si tiene columna time, convertirla
        if 'time' in prep.columns:
            prep['time'] = pd.to_datetime(prep['time'])

        # Filtrar columnas necesarias
        columns_needed = ['cluster', 'outlier_flag']
        if 'time' in prep.columns:
            columns_needed.append('time')
        
        prep_filtered = prep[[col for col in columns_needed if col in prep.columns]].copy()

        # Merge de datos
        if 'time' in prep_filtered.columns:
            df = pd.merge(raw, prep_filtered, on='time', how='inner')
        else:
            raw = raw.reset_index(drop=True)
            prep_filtered = prep_filtered.reset_index(drop=True)
            df = pd.concat([raw, prep_filtered], axis=1)

        # Limpiar nombres de columnas
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

        # Limpiar cluster names
        df['cluster_clean'] = df['cluster'].astype(str).str.replace('cluster_', '', regex=False)
        df['cluster_clean'] = df['cluster_clean'].str.replace('Cluster ', '', regex=False)
        
        # Crear columna de outlier boolean
        df['is_outlier'] = df['outlier_flag'].astype(str).str.lower().isin(['outlier', '1', 'true', 'yes'])
        
        return df
        
    except Exception as e:
        st.error(f"Error cargando datos: {str(e)}")
        return None

@st.cache_data
def load_backtest_results():
    """Carga los resultados de backtesting para comparación"""
    try:
        results = {}
        
        # Cargar resultados con filtro de outliers
        try:
            filtered_df = pd.read_csv('backtesting/optimization_results_filtered.csv')
            results['filtered'] = filtered_df.iloc[0].to_dict()
        except FileNotFoundError:
            results['filtered'] = None
        
        # Cargar resultados sin filtro de outliers  
        try:
            original_df = pd.read_csv('backtesting/optimization_results_original.csv')
            results['original'] = original_df.iloc[0].to_dict()
        except FileNotFoundError:
            results['original'] = None
        
        return results
        
    except Exception as e:
        st.error(f"Error cargando resultados de backtesting: {str(e)}")
        return {'filtered': None, 'original': None}

def get_cluster_colors():
    """Define colores específicos para clusters"""
    return {
        '0': '#1f77b4',  # Azul
        '1': '#ffbb33',  # Amarillo/Naranja
        '2': '#2ca02c',  # Verde
        '3': '#d62728',  # Rojo
        '4': '#9467bd',  # Púrpura
        '5': '#8c564b',  # Marrón
        '6': '#e377c2',  # Rosa
        '7': '#7f7f7f',  # Gris
        '8': '#bcbd22',  # Oliva
        '9': '#17becf'   # Cian
    }

def create_candlestick_chart(df, selected_clusters, show_outliers_only, outlier_display):
    """Crea el gráfico de candlestick con Plotly"""
    
    # Filtrar por clusters seleccionados
    if selected_clusters:
        df_filtered = df[df['cluster_clean'].isin([str(c) for c in selected_clusters])].copy()
    else:
        df_filtered = df.copy()
    
    # Filtrar por outliers si está activado
    if show_outliers_only:
        df_filtered = df_filtered[df_filtered['is_outlier'] == True]
    
    if len(df_filtered) == 0:
        st.warning("No hay datos para mostrar con los filtros seleccionados.")
        return None
    
    # Crear subplot
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Precio EUR/USD', 'Información de Clusters'),
        row_heights=[0.8, 0.2]
    )
    
    cluster_colors = get_cluster_colors()
    unique_clusters = sorted(df_filtered['cluster_clean'].unique())
    
    # Agregar candlestick por cluster
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
        
        # Scatter plot para mostrar clusters en panel inferior
        fig.add_trace(
            go.Scatter(
                x=cluster_data['time'],
                y=[i] * len(cluster_data),
                mode='markers',
                marker=dict(color=color, size=8, symbol='circle'),
                name=f'Cluster {cluster} Timeline',
                showlegend=False,
                hovertemplate=f'<b>Cluster {cluster}</b><br>' +
                             'Tiempo: %{x}<br>' +
                             '<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Agregar marcadores de outliers si están habilitados
    if not show_outliers_only and any(df_filtered['is_outlier']):
        outliers = df_filtered[df_filtered['is_outlier'] == True]
        
        if outlier_display == "Marcadores":
            fig.add_trace(
                go.Scatter(
                    x=outliers['time'],
                    y=outliers['high'] * 1.001,  # Ligeramente arriba del high
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    name='Outliers',
                    hovertemplate='<b>OUTLIER</b><br>' +
                                 'Tiempo: %{x}<br>' +
                                 'Cluster: %{customdata}<br>' +
                                 '<extra></extra>',
                    customdata=outliers['cluster_clean']
                ),
                row=1, col=1
            )
        elif outlier_display == "Líneas verticales":
            for _, outlier in outliers.iterrows():
                fig.add_vline(
                    x=outlier['time'],
                    line=dict(color="red", width=2, dash="dash"),
                    opacity=0.7,
                    row=1, col=1
                )
    
    # Configurar layout
    fig.update_layout(
        title=dict(
            text="📈 Análisis EUR/USD 15M - Trading con Clusters",
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
    
    # Configurar ejes
    fig.update_xaxes(
        title_text="Fecha y Hora",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        row=2, col=1
    )
    
    fig.update_yaxes(
        title_text="Precio",
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
    Recrea un portfolio de vectorbt usando los parámetros de los resultados de backtesting
    """
    if not BACKTESTING_AVAILABLE or backtest_result is None:
        return None
    
    try:
        # Crear modelo con los parámetros optimizados
        ema_period = int(backtest_result['ema_period'])
        use_outlier_filter = bool(backtest_result['use_outlier_filter'])
        
        model = EMAModel(ema_period=ema_period)
        
        # Generar señales
        data_with_signals = model.generate_signals(df_data)
        
        # Desplazar señales para simular ejecución en siguiente vela
        shifted_signals = shift_signals(data_with_signals['signal'])
        
        # Convertir señales a entradas/salidas para vectorbt
        entries = shifted_signals == 1
        exits = shifted_signals == -1
        
        # Forzar cierre de posiciones abiertas para evitar solapamientos
        entries, exits = force_close_open_positions_numba(entries.values, exits.values)
        
        # Convertir de numpy arrays de vuelta a pandas Series para mantener índices
        entries = pd.Series(entries, index=df_data.index)
        exits = pd.Series(exits, index=df_data.index)
        
        # Filtrar entradas en outliers solo si está habilitado
        if use_outlier_filter:
            entries = filter_outlier_entries(entries, df_data['outlier_flag'])
        
        # Ejecutar backtesting con vectorbt
        portfolio = vbt.Portfolio.from_signals(
            df_data['close'],
            entries=entries,
            exits=exits,
            init_cash=1_000_000  # 1M para mejor visualización
        )
        
        return portfolio
        
    except Exception as e:
        st.error(f"Error recreando portfolio: {str(e)}")
        return None

def create_vectorbt_charts(portfolio, title_suffix=""):
    """
    Crea gráficas de vectorbt para órdenes y retornos cumulativos
    """
    if portfolio is None:
        return None, None
    
    try:
        # Gráfica de órdenes (trades)
        orders_fig = go.Figure()
        
        # Agregar precio base
        orders_fig.add_trace(go.Scatter(
            x=portfolio.wrapper.index,
            y=portfolio.close,
            mode='lines',
            name='Precio',
            line=dict(color='black', width=1),
            opacity=0.7
        ))
        
        # Intentar agregar órdenes si están disponibles
        try:
            if hasattr(portfolio, 'orders') and len(portfolio.orders.records_readable) > 0:
                orders_data = portfolio.orders.records_readable
                
                # Verificar si tenemos la columna 'Side' (que vimos en el debug)
                if 'Side' in orders_data.columns:
                    # Usar 'Timestamp' en lugar de 'idx' para el eje X
                    x_column = 'Timestamp' if 'Timestamp' in orders_data.columns else orders_data.index
                    
                    # Agregar órdenes de compra
                    buy_orders = orders_data[orders_data['Side'] == 'Buy']
                    if not buy_orders.empty:
                        orders_fig.add_trace(go.Scatter(
                            x=buy_orders[x_column] if isinstance(x_column, str) else x_column[buy_orders.index],
                            y=buy_orders['Price'],
                            mode='markers',
                            name='Compras',
                            marker=dict(
                                symbol='triangle-up',
                                size=12,
                                color='green',
                                line=dict(width=2, color='darkgreen')
                            ),
                            hovertemplate='<b>Compra</b><br>' +
                                        'Fecha: %{x}<br>' +
                                        'Precio: %{y:.5f}<br>' +
                                        '<extra></extra>'
                        ))
                    
                    # Agregar órdenes de venta
                    sell_orders = orders_data[orders_data['Side'] == 'Sell']
                    if not sell_orders.empty:
                        orders_fig.add_trace(go.Scatter(
                            x=sell_orders[x_column] if isinstance(x_column, str) else x_column[sell_orders.index],
                            y=sell_orders['Price'],
                            mode='markers',
                            name='Ventas',
                            marker=dict(
                                symbol='triangle-down',
                                size=12,
                                color='red',
                                line=dict(width=2, color='darkred')
                            ),
                            hovertemplate='<b>Venta</b><br>' +
                                        'Fecha: %{x}<br>' +
                                        'Precio: %{y:.5f}<br>' +
                                        '<extra></extra>'
                        ))
                    
                    st.success(f"✅ Órdenes cargadas: {len(buy_orders)} compras, {len(sell_orders)} ventas")
                
                elif 'size' in orders_data.columns:
                    # Usar size para determinar dirección
                    x_column = 'Timestamp' if 'Timestamp' in orders_data.columns else orders_data.index
                    
                    buy_orders = orders_data[orders_data['size'] > 0]
                    sell_orders = orders_data[orders_data['size'] < 0]
                    
                    if not buy_orders.empty:
                        orders_fig.add_trace(go.Scatter(
                            x=buy_orders[x_column] if isinstance(x_column, str) else x_column[buy_orders.index],
                            y=buy_orders['Price'],
                            mode='markers',
                            name='Compras',
                            marker=dict(symbol='triangle-up', size=12, color='green')
                        ))
                    
                    if not sell_orders.empty:
                        orders_fig.add_trace(go.Scatter(
                            x=sell_orders[x_column] if isinstance(x_column, str) else x_column[sell_orders.index],
                            y=sell_orders['Price'],
                            mode='markers',
                            name='Ventas',
                            marker=dict(symbol='triangle-down', size=12, color='red')
                        ))
                
                else:
                    # Mostrar todas las órdenes como puntos genéricos
                    x_column = 'Timestamp' if 'Timestamp' in orders_data.columns else orders_data.index
                    orders_fig.add_trace(go.Scatter(
                        x=orders_data[x_column] if isinstance(x_column, str) else x_column,
                        y=orders_data['Price'],
                        mode='markers',
                        name='Órdenes',
                        marker=dict(symbol='circle', size=10, color='blue')
                    ))
            else:
                st.info("No hay órdenes registradas en el portfolio")
        except Exception as e:
            st.warning(f"No se pudieron cargar las órdenes: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        
        orders_fig.update_layout(
            title=f'📊 Órdenes de Trading {title_suffix}',
            xaxis_title='Fecha',
            yaxis_title='Precio',
            height=400,
            showlegend=True
        )
        
        # Gráfica de retornos cumulativos
        cumulative_fig = go.Figure()
        
        # Calcular valor del portfolio a lo largo del tiempo
        portfolio_value = portfolio.value()
        
        cumulative_fig.add_trace(go.Scatter(
            x=portfolio.wrapper.index,
            y=portfolio_value,
            mode='lines',
            name=f'Valor Portfolio {title_suffix}',
            line=dict(width=3),
            fill='tonexty'
        ))
        
        # Agregar línea base del capital inicial
        initial_cash = portfolio.init_cash
        cumulative_fig.add_hline(
            y=initial_cash,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Capital Inicial: ${initial_cash:,.0f}"
        )
        
        cumulative_fig.update_layout(
            title=f'💰 Retornos Cumulativos {title_suffix}',
            xaxis_title='Fecha',
            yaxis_title='Valor del Portfolio ($)',
            yaxis=dict(tickformat='$,.0f'),  # Formato de dinero corregido
            height=400,
            showlegend=True
        )
        
        return orders_fig, cumulative_fig
        
    except Exception as e:
        st.error(f"Error creando gráficas de vectorbt: {str(e)}")
        return None, None

def show_statistics(df, selected_clusters, show_outliers_only):
    """Muestra estadísticas de los datos"""
    
    # Filtrar datos según selección
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
            <h4>📊 Total Velas</h4>
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
            <h4>🎯 Clusters Activos</h4>
            <h2>{}</h2>
        </div>
        """.format(clusters_count), unsafe_allow_html=True)
    
    with col4:
        price_range = df_filtered['high'].max() - df_filtered['low'].min()
        st.markdown("""
        <div class="metric-card">
            <h4>📈 Rango de Precio</h4>
            <h2>{:.5f}</h2>
        </div>
        """.format(price_range), unsafe_allow_html=True)

def show_backtest_comparison(backtest_results):
    """Muestra comparación de resultados de backtesting"""
    
    st.markdown("## 🎯 Comparación de Modelos de Trading")
    st.markdown("### Modelo EMA con y sin filtro de outliers")
    
    filtered_results = backtest_results.get('filtered')
    original_results = backtest_results.get('original')
    
    if not filtered_results and not original_results:
        st.warning("No se encontraron resultados de backtesting. Ejecuta el script train.py primero.")
        return
    
    # Crear columnas para comparación
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("### 🔒 Con Filtro de Outliers")
        if filtered_results:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2ecc71, #27ae60); 
                       color: white; padding: 1.5rem; border-radius: 1rem; margin: 1rem 0;">
                <h4 style="margin: 0 0 1rem 0; color: white;">📊 Métricas Principales</h4>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>EMA Period:</strong> {filtered_results['ema_period']}</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Rendimiento Total:</strong> {filtered_results['total_return']*100:.2f}%</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Max Drawdown:</strong> {filtered_results['max_drawdown']*100:.2f}%</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Sharpe Ratio:</strong> {filtered_results['sharpe_ratio']:.3f}</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Total Trades:</strong> {filtered_results['total_trades']:,}</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Win Rate:</strong> {filtered_results['win_rate']*100:.1f}%</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Valor Objetivo:</strong> {filtered_results['objective_value']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Resultados con filtro no disponibles")
    
    with col2:
        st.markdown("### 🔓 Sin Filtro de Outliers")
        if original_results:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #3498db, #2980b9); 
                       color: white; padding: 1.5rem; border-radius: 1rem; margin: 1rem 0;">
                <h4 style="margin: 0 0 1rem 0; color: white;">📊 Métricas Principales</h4>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>EMA Period:</strong> {original_results['ema_period']}</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Rendimiento Total:</strong> {original_results['total_return']*100:.2f}%</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Max Drawdown:</strong> {original_results['max_drawdown']*100:.2f}%</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Sharpe Ratio:</strong> {original_results['sharpe_ratio']:.3f}</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Total Trades:</strong> {original_results['total_trades']:,}</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Win Rate:</strong> {original_results['win_rate']*100:.1f}%</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Valor Objetivo:</strong> {original_results['objective_value']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Resultados sin filtro no disponibles")
    
    with col3:
        st.markdown("### 📈 Análisis Comparativo")
        if filtered_results and original_results:
            # Calcular diferencias
            return_diff = (filtered_results['total_return'] - original_results['total_return']) * 100
            drawdown_diff = (filtered_results['max_drawdown'] - original_results['max_drawdown']) * 100
            sharpe_diff = filtered_results['sharpe_ratio'] - original_results['sharpe_ratio']
            trades_diff = filtered_results['total_trades'] - original_results['total_trades']
            winrate_diff = (filtered_results['win_rate'] - original_results['win_rate']) * 100
            objective_diff = filtered_results['objective_value'] - original_results['objective_value']
            
            # Determinar colores para las diferencias
            return_color = "green" if return_diff > 0 else "red"
            drawdown_color = "green" if drawdown_diff < 0 else "red"  # Menor drawdown es mejor
            sharpe_color = "green" if sharpe_diff > 0 else "red"
            objective_color = "green" if objective_diff > 0 else "red"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f39c12, #e67e22); 
                       color: white; padding: 1.5rem; border-radius: 1rem; margin: 1rem 0;">
                <h4 style="margin: 0 0 1rem 0; color: white;">🔍 Diferencias (Filtro vs Original)</h4>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Rendimiento:</strong> <span style="color: {return_color};">{return_diff:+.2f}%</span></p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Max Drawdown:</strong> <span style="color: {drawdown_color};">{drawdown_diff:+.2f}%</span></p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Sharpe Ratio:</strong> <span style="color: {sharpe_color};">{sharpe_diff:+.3f}</span></p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Total Trades:</strong> {trades_diff:+,}</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Win Rate:</strong> {winrate_diff:+.1f}%</p>
                <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Valor Objetivo:</strong> <span style="color: {objective_color};">{objective_diff:+.4f}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Conclusión
            st.markdown("### 🎯 Conclusión")
            if objective_diff > 0:
                st.success(f"✅ **El modelo con filtro de outliers es {objective_diff:.2%} mejor** según la métrica objetivo (Rendimiento - Max Drawdown)")
            else:
                st.warning(f"⚠️ **El modelo sin filtro de outliers es {abs(objective_diff):.2%} mejor** según la métrica objetivo (Rendimiento - Max Drawdown)")
        else:
            st.info("Necesitas ambos resultados para comparar")
    
    # Gráfico comparativo
    if filtered_results and original_results:
        st.markdown("### 📊 Visualización Comparativa")
        
        # Crear gráfico de barras comparativo
        metrics = ['Rendimiento Total (%)', 'Max Drawdown (%)', 'Sharpe Ratio', 'Win Rate (%)']
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
            name='Con Filtro de Outliers',
            x=metrics,
            y=filtered_values,
            marker_color='#2ecc71',
            text=[f'{v:.2f}' for v in filtered_values],
            textposition='auto'
        ))
        
        fig_comparison.add_trace(go.Bar(
            name='Sin Filtro de Outliers',
            x=metrics,
            y=original_values,
            marker_color='#3498db',
            text=[f'{v:.2f}' for v in original_values],
            textposition='auto'
        ))
        
        fig_comparison.update_layout(
            title='📊 Comparación de Métricas de Trading',
            xaxis_title='Métricas',
            yaxis_title='Valores',
            barmode='group',
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Gráficas de vectorbt
        if BACKTESTING_AVAILABLE and filtered_results and original_results:
            st.markdown("### 📈 Análisis Detallado de Trading")
            
            # Cargar datos para recrear portfolios
            try:
                data_path = Path(__file__).parent / "data_preprocessed" / "enhanced_eurusd_dataset.csv"
                if data_path.exists():
                    trading_data = load_backtest_data(str(data_path))
                    
                    # Usar solo una muestra para visualización (últimos 10k registros)
                    sample_size = min(10000, len(trading_data))
                    trading_data = trading_data.iloc[-sample_size:].copy()
                    
                    st.info(f"📊 Generando gráficas con {len(trading_data):,} registros (últimos datos)")
                    
                    # Recrear portfolios
                    with st.spinner("Recreando portfolios..."):
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
                    
                    # Comparación de retornos cumulativos en una sola gráfica
                    if portfolio_filtered is not None and portfolio_original is not None:
                        st.markdown("#### 📊 Comparación Directa de Retornos")
                        
                        comparison_returns_fig = go.Figure()
                        
                        # Portfolio con filtro
                        comparison_returns_fig.add_trace(go.Scatter(
                            x=portfolio_filtered.wrapper.index,
                            y=portfolio_filtered.value(),
                            mode='lines',
                            name='Con Filtro de Outliers',
                            line=dict(color='#2ecc71', width=3)
                        ))
                        
                        # Portfolio sin filtro
                        comparison_returns_fig.add_trace(go.Scatter(
                            x=portfolio_original.wrapper.index,
                            y=portfolio_original.value(),
                            mode='lines',
                            name='Sin Filtro de Outliers',
                            line=dict(color='#3498db', width=3)
                        ))
                        
                        # Línea base del capital inicial
                        initial_cash = portfolio_filtered.init_cash
                        comparison_returns_fig.add_hline(
                            y=initial_cash,
                            line_dash="dash",
                            line_color="gray",
                            annotation_text=f"Capital Inicial: ${initial_cash:,.0f}"
                        )
                        
                        comparison_returns_fig.update_layout(
                            title='💰 Comparación de Retornos Cumulativos',
                            xaxis_title='Fecha',
                            yaxis_title='Valor del Portfolio ($)',
                            yaxis=dict(tickformat='$,.0f'),  # Formato de dinero corregido
                            height=500,
                            showlegend=True,
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )
                        
                        st.plotly_chart(comparison_returns_fig, use_container_width=True)
                
                else:
                    st.warning("No se encontró el archivo de datos preprocesados para generar las gráficas de trading")
                    
            except Exception as e:
                st.error(f"Error generando gráficas de trading: {str(e)}")
        
        elif not BACKTESTING_AVAILABLE:
            st.info("Las gráficas detalladas de trading requieren los módulos de backtesting.")

def main():
    # Título principal
    st.title("🚀 EUR/USD Trading Analysis Dashboard")
    st.markdown("### Análisis avanzado con clustering y detección de outliers")
    
    # Cargar datos
    with st.spinner("Cargando datos..."):
        df = load_data()
        backtest_results = load_backtest_results()
    
    if df is None:
        st.stop()
    
    # Navegación por pestañas
    tab1, tab2 = st.tabs(["📈 Análisis de Datos", "🎯 Resultados de Backtesting"])
    
    with tab1:
        # Sidebar para controles
        st.sidebar.header("⚙️ Configuración")
        
        # Información general en sidebar
        st.sidebar.markdown(f"""
        **📋 Resumen de Datos:**
        - **Total de velas:** {len(df):,}
        - **Período:** {df['time'].min().strftime('%Y-%m-%d')} → {df['time'].max().strftime('%Y-%m-%d')}
        - **Clusters disponibles:** {', '.join(sorted(df['cluster_clean'].unique()))}
        - **Outliers totales:** {df['is_outlier'].sum():,} ({df['is_outlier'].sum()/len(df)*100:.1f}%)
        """)
        
        st.sidebar.markdown("---")
        
        # Control de clusters
        available_clusters = sorted(df['cluster_clean'].unique())
        selected_clusters = st.sidebar.multiselect(
            "🎯 Seleccionar Clusters:",
            options=available_clusters,
            default=available_clusters,
            help="Selecciona qué clusters mostrar en el gráfico"
        )
        
        # Control de outliers
        show_outliers_only = st.sidebar.checkbox(
            "🚨 Mostrar solo Outliers",
            value=False,
            help="Cuando está activado, solo muestra las velas marcadas como outliers"
        )
        
        # Estilo de outliers (solo si no está en modo "solo outliers")
        if not show_outliers_only:
            outlier_display = st.sidebar.selectbox(
                "🎨 Estilo de Outliers:",
                options=["Marcadores", "Líneas verticales", "Sin mostrar"],
                index=0,
                help="Cómo mostrar los outliers en el gráfico"
            )
        else:
            outlier_display = "Sin mostrar"
        
        # Control de rango de datos
        st.sidebar.markdown("---")
        st.sidebar.subheader("📅 Rango de Datos")
        
        # Slider para seleccionar rango de datos
        date_range = st.sidebar.slider(
            "Seleccionar rango:",
            min_value=0,
            max_value=len(df)-1,
            value=(0, min(1000, len(df)-1)),
            help="Desliza para seleccionar el rango de velas a mostrar"
        )
        
        # Filtrar por rango de fechas
        df_range = df.iloc[date_range[0]:date_range[1]+1].copy()
        
        # Mostrar estadísticas
        st.markdown("## 📊 Estadísticas")
        show_statistics(df_range, selected_clusters, show_outliers_only)
        
        # Información de clusters
        if selected_clusters:
            st.markdown("## 🎯 Información de Clusters Seleccionados")
            
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
                            <strong>Velas:</strong> {len(cluster_data):,}<br>
                            <strong>Outliers:</strong> {outliers_in_cluster} ({outliers_pct:.1f}%)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Crear y mostrar gráfico
        st.markdown("## 📈 Gráfico de Candlestick")
        
        fig = create_candlestick_chart(df_range, selected_clusters, show_outliers_only, outlier_display)
        
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            
            # Información adicional
            with st.expander("ℹ️ Información adicional"):
                st.markdown("""
                **Cómo usar este dashboard:**
                
                1. **Clusters:** Utiliza el selector múltiple en la barra lateral para elegir qué clusters mostrar
                2. **Outliers:** Activa "Mostrar solo Outliers" para ver únicamente las velas anómalas
                3. **Estilos:** Cambia cómo se muestran los outliers (marcadores, líneas, o sin mostrar)
                4. **Rango:** Ajusta el slider para navegar por diferentes períodos de tiempo
                
                **Colores de Clusters:**
                - 🔵 Cluster 0: Azul
                - 🟡 Cluster 1: Amarillo/Naranja
                - 🟢 Cluster 2+: Otros colores
                
                **Marcadores:**
                - 🔺 Triángulos rojos: Outliers detectados
                - 📊 Panel inferior: Timeline de clusters
                """)
    
    with tab2:
        # Mostrar resultados de backtesting
        show_backtest_comparison(backtest_results)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        💡 Dashboard creado con Streamlit y Plotly | 
        📈 Análisis de EUR/USD 15M con Machine Learning
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        get_script_run_ctx = lambda: None

    ctx = get_script_run_ctx()

    if ctx is None:
        print("Esta aplicación debe ejecutarse con 'streamlit run app.py'.")
    else:
        main()