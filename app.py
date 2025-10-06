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

def main():
    # Título principal
    st.title("🚀 EUR/USD Trading Analysis Dashboard")
    st.markdown("### Análisis avanzado con clustering y detección de outliers")
    
    # Cargar datos
    with st.spinner("Cargando datos..."):
        df = load_data()
    
    if df is None:
        st.stop()
    
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