# main.py
"""
RedShield AI: Phoenix v4.0 - Plataforma Proactiva de Respuesta a Emergencias y Asignación de Recursos

Este es el punto de entrada principal para la aplicación Streamlit. Orquesta la
interfaz de usuario, la gestión de datos y el motor de análisis predictivo para ofrecer
un panel de control interactivo y en tiempo real para el personal de mando de respuesta a emergencias.

La aplicación se estructura en tres pestañas principales:
1. Mando Operativo: Un mapa en vivo y medidores de soporte a la decisión para acción inmediata.
2. Análisis Profundo de KPIs: Analíticas y visualizaciones avanzadas para una comprensión profunda.
3. Metodología y Perspectivas: Una explicación detallada de los modelos subyacentes.
"""

import logging
import warnings
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# --- Configuración del Sistema: DEBE ser el primer comando de Streamlit ---
st.set_page_config(
    page_title="RedShield AI: Phoenix v4.0",
    layout="wide",
    initial_sidebar_state="expanded"
)

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# --- Importación de módulos refactorizados (DESPUÉS de st.set_page_config) ---
from core import DataManager, EnvFactors, PredictiveAnalyticsEngine
from utils import ReportGenerator, load_config

# --- Constantes de la Aplicación Centralizadas ---
CONSTANTS = {
    'RISK_COVERAGE_PER_UNIT': 0.25,
    'PRESSURE_WEIGHTS': {'traffic': 0.3, 'hospital': 0.4, 'adequacy': 0.3},
    'TRAFFIC_MIN': 0.5,
    'TRAFFIC_MAX': 3.0,
    'FALLBACK_POP_DENSITY': 50000,
    'FLOAT_TOLERANCE': 1e-6
}

# --- Configuración Post-Página ---
st.cache_data.clear()
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Configuración de Logging ---
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/redshield_phoenix.log")
    ]
)
logger = logging.getLogger(__name__)


# --- Clase de Datos Personalizada para Reruns Eficientes ---
class EnvFactorsWithTolerance(EnvFactors):
    """
    Extiende EnvFactors para incluir una comprobación de igualdad personalizada con tolerancia de flotantes.
    Esto previene reruns innecesarios de Streamlit por ruido menor en punto flotante.
    """
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EnvFactors): return NotImplemented
        for attr in ['traffic_level', 'air_quality_index', 'hospital_divert_status']:
            if abs(getattr(self, attr, 0) - getattr(other, attr, 0)) > CONSTANTS['FLOAT_TOLERANCE']: return False
        for attr in ['is_holiday', 'weather', 'major_event', 'heatwave_alert', 'day_type', 'time_of_day', 'public_event_type', 'police_activity', 'school_in_session']:
            if getattr(self, attr, None) != getattr(other, attr, None): return False
        return True


# --- Clase Principal del Dashboard ---
class Dashboard:
    """Maneja el renderizado de la interfaz de usuario de Streamlit para Phoenix v4.0."""
    def __init__(self, dm: DataManager, engine: PredictiveAnalyticsEngine):
        self.dm = dm
        self.engine = engine
        self.config = dm.config
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Inicializa todas las claves de estado de sesión requeridas si no existen."""
        if 'avg_pop_density' not in st.session_state:
            if not self.dm.zones_gdf.empty and 'population' in self.dm.zones_gdf.columns:
                st.session_state['avg_pop_density'] = self.dm.zones_gdf['population'].mean()
            else:
                st.session_state['avg_pop_density'] = CONSTANTS['FALLBACK_POP_DENSITY']
        
        # Define los estados iniciales en español
        default_states = {
            'historical_data': [], 'kpi_df': pd.DataFrame(), 'forecast_df': pd.DataFrame(),
            'allocations': {}, 'sparkline_data': {}, 'current_incidents': [],
            'simulation_mode': False,
            'sim_incident_count': 0,
            'sim_ambulance_count': 0,
            'env_factors': EnvFactorsWithTolerance(
                is_holiday=False, weather="Despejado", traffic_level=1.0, major_event=False,
                population_density=st.session_state.avg_pop_density, air_quality_index=50.0,
                heatwave_alert=False, day_type='Entre Semana', time_of_day='Mediodía',
                public_event_type='Ninguno', hospital_divert_status=0.0,
                police_activity='Normal', school_in_session=True)}
        
        for key, value in default_states.items():
            if key not in st.session_state: st.session_state[key] = value

    def render(self):
        """Método principal de renderizado para todo el dashboard."""
        st.title("RedShield AI: Phoenix v4.0")
        st.markdown("##### Plataforma Proactiva de Respuesta a Emergencias y Asignación de Recursos")
        self._render_sidebar()

        with st.spinner("Ejecutando Análisis Avanzado y Optimización..."):
            self._run_analytics_pipeline()

        tab1, tab2, tab3 = st.tabs(["🔥 Mando Operativo", "📊 Análisis Profundo de KPIs", "🧠 Metodología"])
        with tab1: self._render_operational_command_tab()
        with tab2: self._render_kpi_deep_dive_tab()
        with tab3: self._render_methodology_tab()

    def _run_analytics_pipeline(self):
        """Ejecuta la tubería de análisis usando la fuente de datos correcta (en vivo o simulada)."""
        try:
            if not st.session_state.simulation_mode:
                st.session_state.current_incidents = self.dm.get_current_incidents(st.session_state.env_factors)
                self.dm.ambulances = self.dm._initialize_ambulances()

            kpi_df, spark_data = self.engine.generate_kpis_with_sparklines(
                st.session_state.historical_data, st.session_state.env_factors, st.session_state.current_incidents
            )
            forecast_df = self.engine.generate_forecast(kpi_df)
            allocations = self.engine.generate_allocation_recommendations(kpi_df)
            
            st.session_state.update({
                'kpi_df': kpi_df, 'forecast_df': forecast_df, 'allocations': allocations,
                'sparkline_data': spark_data
            })
        except Exception as e:
            logger.error(f"Fallo en la tubería de análisis: {e}", exc_info=True)
            st.error(f"Error en la tubería de análisis: {e}. Revise los registros para más detalles.")
            st.session_state.update({'kpi_df':pd.DataFrame(),'forecast_df':pd.DataFrame(),'allocations':{},'sparkline_data':{}})

    # --- PESTAÑA 1: MANDO OPERATIVO ---
    
    def _render_operational_command_tab(self):
        """Renderiza la vista principal de mando operativo."""
        self._render_system_status_bar()
        
        with st.expander("Mostrar Detalles de Tendencia en el Sistema de Urgencias"):
            self._render_sparkline_details()
        
        st.divider()
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("Mapa de Operaciones en Tiempo Real")
            map_data = self._prepare_map_data(kpi_df=st.session_state.kpi_df, _zones_gdf=self.dm.zones_gdf)
            if map_data is not None:
                map_object = self._render_dynamic_map(map_gdf=map_data, incidents=st.session_state.current_incidents, _ambulances=self.dm.ambulances)
                if map_object: st_folium(map_object, use_container_width=True, height=600)
        with col2:
            st.subheader("Soporte a la Toma de Decisiones")
            self._plot_system_pressure_gauge()
            self._plot_resource_to_risk_adequacy()
            
        st.divider()
        self._render_key_risk_indicators(st.session_state.kpi_df)

    @staticmethod
    def _create_status_metric(label, value, trend, color): return f'<div style="flex:1;background-color:{color};padding:10px;text-align:center;color:white;border-right:1px solid #fff4;"><div style="font-size:1.5rem;font-weight:bold;">{value} {trend}</div><div style="font-size:0.8rem;">{label}</div></div>'

    def _render_system_status_bar(self):
        st.subheader("Estado de 'Salud' de la Respuesta en el Sistema")
        kpi_df, spark_data = st.session_state.kpi_df, st.session_state.sparkline_data
        if kpi_df.empty or not spark_data: st.info("Estado del sistema no disponible..."); return
        try:
            inc_val, amb_val = len(st.session_state.current_incidents), sum(1 for a in self.dm.ambulances.values() if a['status'] == 'Disponible')
            risk_val, adeq_val = kpi_df['Integrated_Risk_Score'].max(), kpi_df['Resource Adequacy Index'].mean()
            inc_data, amb_data, risk_data, adeq_data = (spark_data.get('active_incidents',{'values':[inc_val],'range':[inc_val-1,inc_val+1]}), spark_data.get('available_ambulances',{'values':[amb_val],'range':[amb_val-1,amb_val+1]}), spark_data.get('max_risk',{'values':[risk_val],'range':[0,1]}), spark_data.get('adequacy',{'values':[adeq_val],'range':[0,1]}))
            def get_color(v, r, h): return "#D32F2F" if (h and v>r[1])or(not h and v<r[0])else "#FBC02D" if (h and v>r[0])or(not h and v<r[1])else "#388E3C"
            def get_trend(d): return "▲" if len(d)>1 and d[-1]>d[-2] else "▼" if len(d)>1 and d[-1]<d[-2] else "▬"
            metrics = [self._create_status_metric("Incidentes Activos",f"{inc_val}",get_trend(inc_data['values']),get_color(inc_val,inc_data['range'],True)), self._create_status_metric("Unidades Disponibles",f"{amb_val}",get_trend(amb_data['values']),get_color(amb_val,amb_data['range'],False)), self._create_status_metric("Riesgo Máx. de Zona",f"{risk_val:.3f}",get_trend(risk_data['values']),get_color(risk_val,risk_data['range'],True)), self._create_status_metric("Suficiencia del Sistema",f"{adeq_val:.1%}",get_trend(adeq_data['values']),get_color(adeq_val,adeq_data['range'],False))]
            metrics[-1] = metrics[-1].replace('border-right: 1px solid #fff4;','')
            st.markdown(f'<div style="display:flex;border:1px solid #444;border-radius:5px;overflow:hidden;font-family:sans-serif;">{"".join(metrics)}</div>',True)
        except Exception as e:
            logger.error(f"Error al renderizar la barra de estado: {e}", exc_info=True); st.warning(f"No se pudo renderizar la barra de estado del sistema: {e}")

    @st.cache_data
    def _prepare_map_data(_self, kpi_df: pd.DataFrame, _zones_gdf: gpd.GeoDataFrame):
        if _zones_gdf.empty or kpi_df.empty: return None
        map_gdf = _zones_gdf.join(kpi_df.set_index('Zone'), on='name')
        return map_gdf.reset_index()

    def _render_dynamic_map(self, map_gdf: gpd.GeoDataFrame, incidents: List[Dict], _ambulances: Dict):
        try:
            center = map_gdf.unary_union.centroid
            m = folium.Map(location=[center.y, center.x], zoom_start=11, tiles="cartodbpositron", prefer_canvas=True)
            folium.Choropleth(geo_data=map_gdf, data=map_gdf, columns=['name','Integrated_Risk_Score'], key_on='feature.properties.name', fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.2, legend_name='Puntaje de Riesgo Integrado', name="Mapa de Calor de Riesgo").add_to(m)
            inc_fg = MarkerCluster(name='Incidentes en Vivo', show=True).add_to(m)
            for inc in incidents:
                if (loc:=inc.get('location')) and 'lat' in loc and 'lon' in loc:
                    folium.Marker([loc['lat'], loc['lon']], tooltip=f"Tipo: {inc.get('type','N/A')}<br>Triage: {inc.get('triage','N/A')}", icon=folium.Icon(color='red', icon="car-crash" if "Accident" in inc.get('type','') else "first-aid", prefix='fa')).add_to(inc_fg)
            amb_fg = folium.FeatureGroup(name='Alcance de Unidades (5-min)', show=False).add_to(m)
            for amb_id, amb_data in _ambulances.items():
                if amb_data.get('status') == 'Disponible':
                    loc = amb_data.get('location'); folium.Circle([loc.y, loc.x], radius=2400, color='#1E90FF', fill=True, fill_opacity=0.1, tooltip=f"Alcance de Unidad {amb_id}").add_to(amb_fg)
                    folium.Marker([loc.y, loc.x], icon=folium.Icon(color='blue',icon='ambulance',prefix='fa'), tooltip=f"Unidad {amb_id} (Disponible)").add_to(amb_fg)
            folium.LayerControl().add_to(m); return m
        except Exception as e:
            logger.error(f"Error al renderizar el mapa: {e}", exc_info=True); st.error(f"Error al renderizar el mapa: {e}"); return None

    def _plot_system_pressure_gauge(self):
        try:
            kpi_df, env = st.session_state.kpi_df, st.session_state.env_factors
            if kpi_df.empty: return
            t_norm = np.clip((env.traffic_level-CONSTANTS['TRAFFIC_MIN'])/(CONSTANTS['TRAFFIC_MAX']-CONSTANTS['TRAFFIC_MIN']),0,1)
            h_norm, a_norm = env.hospital_divert_status, 1-kpi_df['Resource Adequacy Index'].mean()
            score = (t_norm*CONSTANTS['PRESSURE_WEIGHTS']['traffic']+h_norm*CONSTANTS['PRESSURE_WEIGHTS']['hospital']+a_norm*CONSTANTS['PRESSURE_WEIGHTS']['adequacy'])*125
            fig = go.Figure(go.Indicator(mode="gauge+number", value=min(score,100), title={'text':"Presión del Sistema"}, gauge={'axis':{'range':[0,100]}, 'bar':{'color':"#222"}, 'steps':[{'range':[0,40],'color':'#388E3C'},{'range':[40,75],'color':'#FBC02D'},{'range':[75,100],'color':'#D32F2F'}]}))
            fig.update_layout(height=250, margin=dict(l=10,r=10,t=40,b=10)); st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error en medidor de presión: {e}", exc_info=True); st.warning("No se pudo mostrar el medidor de Presión del Sistema.")

    def _plot_resource_to_risk_adequacy(self):
        try:
            kpi_df, allocations = st.session_state.kpi_df, st.session_state.allocations
            if kpi_df.empty or 'Integrated_Risk_Score' not in kpi_df.columns: st.info("No hay datos para el gráfico de adecuación de riesgo."); return
            df = kpi_df.nlargest(7,'Integrated_Risk_Score').copy(); df['allocated']=df['Zone'].map(allocations).fillna(0)
            risk_cov = self.config.get('model_params',{}).get('risk_coverage_per_unit',CONSTANTS['RISK_COVERAGE_PER_UNIT'])
            df['risk_covered']=df['allocated']*risk_cov; df['ratio']=np.clip((df['risk_covered']+1e-9)/(df['Integrated_Risk_Score']+1e-9),0,1.5)
            df['color']=df['ratio'].apply(lambda r: '#D32F2F' if r<0.7 else '#FBC02D' if r<1.0 else '#388E3C')
            fig=go.Figure(); fig.add_trace(go.Bar(y=df['Zone'],x=df['Integrated_Risk_Score'],orientation='h',name='Riesgo Total (Demanda)',marker_color='#e0e0e0',hovertemplate="<b>Zona:</b> %{y}<br><b>Riesgo Total:</b> %{x:.3f}<extra></extra>"))
            fig.add_trace(go.Bar(y=df['Zone'],x=df['risk_covered'],orientation='h',name='Riesgo Cubierto (Oferta)',marker_color=df['color'],text=df['allocated'].astype(int).astype(str)+" Unidad(es)",textposition='inside',textfont=dict(color='white',size=12),hovertemplate="<b>Zona:</b> %{y}<br><b>Riesgo Cubierto:</b> %{x:.3f}<br><b>Asignado:</b> %{text}<extra></extra>"))
            fig.update_layout(title='Recursos vs. Demanda para Zonas de Alto Riesgo',xaxis_title='Puntaje de Riesgo Integrado',yaxis_title=None,height=350,yaxis={'categoryorder':'total ascending'},legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),barmode='overlay',plot_bgcolor='white',margin=dict(l=10,r=10,t=70,b=10))
            st.plotly_chart(fig, use_container_width=True); st.markdown("**Cómo leer:** La barra gris es el riesgo (demanda). La barra de color es la cobertura (oferta).")
        except Exception as e:
            logger.error(f"Error en gráfico de adecuación: {e}", exc_info=True); st.warning("No se pudo mostrar el gráfico de adecuación de recursos.")

    def _render_sparkline_details(self):
        """Renderiza los gráficos de chispa detallados en un diseño de 4 columnas."""
        kpi_df = st.session_state.kpi_df; spark_data = st.session_state.sparkline_data
        if kpi_df.empty or not spark_data: st.info("Datos de tendencia aún no disponibles."); return
        try:
            inc_val, amb_val = len(st.session_state.current_incidents), sum(1 for a in self.dm.ambulances.values() if a['status'] == 'Disponible')
            risk_val, adeq_val = kpi_df['Integrated_Risk_Score'].max(), kpi_df['Resource Adequacy Index'].mean()
            inc_data, amb_data, risk_data, adeq_data = (spark_data.get('active_incidents',{'values':[inc_val]*5,'range':[inc_val-1,inc_val+1]}), spark_data.get('available_ambulances',{'values':[amb_val]*5,'range':[amb_val-1,amb_val+1]}), spark_data.get('max_risk',{'values':[risk_val]*5,'range':[0,1]}), spark_data.get('adequacy',{'values':[adeq_val]*5,'range':[0,1]}))
            def get_color(v, r, h=True): return "#D32F2F" if (h and v>r[1])or(not h and v<r[0])else "#FBC02D" if (h and v>r[0])or(not h and v<r[1])else "#388E3C"
            cols = st.columns(4)
            with cols[0]: st.plotly_chart(self._create_sparkline_plot(inc_data['values'], inc_data['range'], f"{inc_val}", "Incidentes Activos", get_color(inc_val,inc_data['range'])), use_container_width=True)
            with cols[1]: st.plotly_chart(self._create_sparkline_plot(amb_data['values'], amb_data['range'], f"{amb_val}", "Unidades Disponibles", get_color(amb_val,amb_data['range'],False)), use_container_width=True)
            with cols[2]: st.plotly_chart(self._create_sparkline_plot(risk_data['values'], risk_data['range'], f"{risk_val:.3f}", "Riesgo Máx. de Zona", get_color(risk_val,risk_data['range'])), use_container_width=True)
            with cols[3]: st.plotly_chart(self._create_sparkline_plot(adeq_data['values'], adeq_data['range'], f"{adeq_val:.1%}", "Suficiencia del Sistema", get_color(adeq_val,adeq_data['range'],False)), use_container_width=True)
        except Exception as e:
            logger.error(f"Error al renderizar gráficos de chispa: {e}", exc_info=True); st.warning("No se pudieron mostrar los detalles de tendencia.")

    def _render_key_risk_indicators(self, kpi_df: pd.DataFrame):
        st.subheader("Perfiles de Riesgos Clave")
        st.markdown("Un resumen visual de las zonas de mayor riesgo, mostrando la magnitud total del riesgo y la composición de sus impulsores clave.")
        try:
            req = ['Zone','Integrated_Risk_Score','Violence Clustering Score','Medical Surge Score','Spatial Spillover Risk']
            if not all(c in kpi_df.columns for c in req): st.error("Faltan datos para los Perfiles de Riesgo Clave."); logger.warning(f"Faltan columnas para KRI: {set(req)-set(kpi_df.columns)}"); return
            df_top = kpi_df.nlargest(5, 'Integrated_Risk_Score').copy()
            if df_top.empty: st.info("No hay zonas con riesgo significativo para mostrar perfiles."); return
            colors = {"Violence":"#D32F2F", "Medical":"#1E90FF", "Spillover":"#FF9800"}
            for i, row in df_top.iterrows():
                st.markdown("---"); c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f"##### {row['Zone']}")
                    st.progress(row['Integrated_Risk_Score'], text=f"Riesgo Integrado: {row['Integrated_Risk_Score']:.3f}")
                with c2:
                    sc = st.columns(3)
                    with sc[0]:
                        score, color = row['Violence Clustering Score'], colors['Violence']
                        st.markdown(f"<p style='color:{color};font-size:13px;margin-bottom:-10px;'><b>Violencia</b></p>", unsafe_allow_html=True)
                        fig = go.Figure(go.Indicator(mode="gauge+number",value=score*100,number={'font':{'size':24,'color':color}},gauge={'axis':{'range':[None,100]},'bar':{'color':color,'thickness':0.8}}))
                        fig.update_layout(height=80,margin=dict(l=0,r=0,t=20,b=0),paper_bgcolor='rgba(0,0,0,0)'); st.plotly_chart(fig,use_container_width=True)
                    with sc[1]:
                        score, color = row['Medical Surge Score'], colors['Medical']
                        st.markdown(f"<p style='color:{color};font-size:13px;margin-bottom:-10px;'><b>Médico</b></p>", unsafe_allow_html=True)
                        fig = go.Figure(go.Indicator(mode="gauge+number",value=score*100,number={'font':{'size':24,'color':color}},gauge={'axis':{'range':[None,100]},'bar':{'color':color,'thickness':0.8}}))
                        fig.update_layout(height=80,margin=dict(l=0,r=0,t=20,b=0),paper_bgcolor='rgba(0,0,0,0)'); st.plotly_chart(fig,use_container_width=True)
                    with sc[2]:
                        score, color = row['Spatial Spillover Risk'], colors['Spillover']
                        st.markdown(f"<p style='color:{color};font-size:13px;margin-bottom:-10px;'><b>Desbordamiento</b></p>", unsafe_allow_html=True)
                        fig = go.Figure(go.Indicator(mode="gauge+number",value=score*100,number={'font':{'size':24,'color':color}},gauge={'axis':{'range':[None,100]},'bar':{'color':color,'thickness':0.8}}))
                        fig.update_layout(height=80,margin=dict(l=0,r=0,t=20,b=0),paper_bgcolor='rgba(0,0,0,0)'); st.plotly_chart(fig,use_container_width=True)
        except Exception as e:
            logger.error(f"Error al renderizar Indicadores Clave de Riesgo: {e}", exc_info=True); st.warning("No se pudieron mostrar los Perfiles de Riesgo Clave.")

    # --- PESTAÑA 2: ANÁLISIS DE KPIs ---
    def _render_kpi_deep_dive_tab(self):
        st.subheader("Matriz Comprensiva de Indicadores de Riesgo")
        kpi_df = st.session_state.kpi_df
        if not kpi_df.empty:
            st.dataframe(kpi_df.set_index('Zone').style.format("{:.3f}").background_gradient(cmap='viridis'), use_container_width=True)
        else: st.info("Datos de KPI aún no disponibles.")
        st.divider()
        st.subheader("Visualizaciones Analíticas Avanzadas")
        if not kpi_df.empty:
            tab_titles = ["📍 Vista Estratégica", "🎯 Asignación de Recursos", "⏱️ Tendencias del Riesgo", "🧬 Anatomía de las Zonas Críticas", "🧩 Análisis Detallado de Zonas", "🔭 Pronóstico de Incidentes a 72 Horas"]
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_titles)
            with tab1: self._plot_vulnerability_quadrant(kpi_df)
            with tab2: self._plot_allocation_opportunity(kpi_df, st.session_state.allocations)
            with tab3: self._plot_risk_momentum(kpi_df)
            with tab4: self._plot_critical_zone_anatomy(kpi_df)
            with tab5: self._plot_risk_contribution_sunburst(kpi_df)
            with tab6: self._plot_forecast_with_uncertainty()
        else: st.info("Visualizaciones avanzadas no disponibles: esperando datos...")

    @staticmethod
    def _create_sparkline_plot(data, normal_range, current_value_text, label, color, high_is_bad=True):
        fig = go.Figure()
        fig.add_shape(type="rect", xref="x", yref="y", x0=0, y0=normal_range[0], x1=len(data)-1, y1=normal_range[1], fillcolor="#388E3C", opacity=0.15, layer="below", line_width=0)
        fig.add_trace(go.Scatter(x=list(range(len(data))), y=data, mode='lines', line=dict(color=color, width=3), hoverinfo='none'))
        fig.add_trace(go.Scatter(x=[len(data)-1], y=[data[-1]], mode='markers', marker=dict(color=color, size=10, line=dict(width=2, color='white')), hoverinfo='none'))
        fig.add_annotation(xref="paper", yref="paper", x=0.02, y=0.98, text=f"<b>{current_value_text}</b>", showarrow=False, font=dict(size=28, color=color, family="Arial Black, sans-serif"), align="left", xanchor="left", yanchor="top")
        fig.add_annotation(xref="paper", yref="paper", x=0.98, y=0.05, text=label, showarrow=False, font=dict(size=14, color="#666"), align="right", xanchor="right", yanchor="bottom")
        plot_min, plot_max = min(min(data), normal_range[0]), max(max(data), normal_range[1])
        padding = (plot_max - plot_min) * 0.15
        fig.update_layout(yaxis=dict(range=[plot_min-padding, plot_max+padding], showticklabels=True, tickfont=dict(size=10, color="#999"), side='right', nticks=4, showgrid=False), xaxis=dict(visible=False), showlegend=False, plot_bgcolor='rgba(240,240,240,0.95)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=5,r=40,t=5,b=5), height=120)
        return fig

    def _plot_vulnerability_quadrant(self, kpi_df: pd.DataFrame):
        st.markdown("**Análisis:** Segmenta las zonas por **vulnerabilidad estructural** (ej. centralidad en la red vial) vs. **riesgo dinámico** inmediato (ej. incidentes recientes, eventos).")
        try:
            req = ['Ensemble Risk Score','GNN_Structural_Risk','Integrated_Risk_Score','Expected Incident Volume']
            if not all(c in kpi_df.columns for c in req): st.error("Faltan datos para la Matriz de Riesgo."); return
            x_threshold, y_threshold = kpi_df['Ensemble Risk Score'].quantile(0.75), kpi_df['GNN_Structural_Risk'].quantile(0.75)
            max_incidents = max(kpi_df['Expected Incident Volume'].max(),1); sizeref = 2.*max_incidents/(40.**2)
            fig = go.Figure()
            x_max, y_max = kpi_df['Ensemble Risk Score'].max()*1.1, kpi_df['GNN_Structural_Risk'].max()*1.1
            fig.add_shape(type="rect",xref="x",yref="y",x0=x_threshold,y0=y_threshold,x1=x_max,y1=y_max,fillcolor="rgba(229,57,53,0.07)",line_width=0,layer="below")
            fig.add_shape(type="rect",xref="x",yref="y",x0=x_threshold,y0=0,x1=x_max,y1=y_threshold,fillcolor="rgba(255,179,0,0.07)",line_width=0,layer="below")
            fig.add_shape(type="rect",xref="x",yref="y",x0=0,y0=y_threshold,x1=x_threshold,y1=y_max,fillcolor="rgba(25,118,210,0.07)",line_width=0,layer="below")
            fig.add_trace(go.Scatter(x=kpi_df['Ensemble Risk Score'],y=kpi_df['GNN_Structural_Risk'],mode='markers',marker=dict(size=kpi_df['Expected Incident Volume'],sizemode='area',sizeref=sizeref,sizemin=6,color=kpi_df['Integrated_Risk_Score'],colorscale='Plasma',showscale=True,colorbar=dict(title='Riesgo<br>Total',x=1.15,thickness=20,tickfont=dict(size=10)),line=dict(width=1,color='DarkSlateGrey')),customdata=kpi_df['Zone'],hovertemplate="<b>Zona: %{customdata}</b><br>Riesgo Dinámico: %{x:.3f}<br>Riesgo Estructural: %{y:.3f}<extra></extra>"))
            crisis_zones = kpi_df[(kpi_df['Ensemble Risk Score']>=x_threshold)&(kpi_df['GNN_Structural_Risk']>=y_threshold)]
            if not crisis_zones.empty:
                fig.add_trace(go.Scatter(x=crisis_zones['Ensemble Risk Score'],y=crisis_zones['GNN_Structural_Risk'],mode='text',text=crisis_zones['Zone'],textposition="top center",textfont=dict(size=10,color="#333",family="Arial Black"),showlegend=False,hoverinfo='none'))
            fig.add_vline(x=x_threshold,line_width=1.5,line_dash="longdash",line_color="rgba(0,0,0,0.2)");fig.add_hline(y=y_threshold,line_width=1.5,line_dash="longdash",line_color="rgba(0,0,0,0.2)")
            anno_defaults = dict(xref="paper",yref="paper",showarrow=False,font=dict(family="Arial,sans-serif",size=11,color="rgba(0,0,0,0.5)"))
            fig.add_annotation(x=0.98,y=0.98,text="<b>ZONAS DE CRISIS</b><br>(Alto Dinámico, Alta Estructural)",xanchor='right',yanchor='top',align='right',**anno_defaults)
            fig.add_annotation(x=0.98,y=0.02,text="<b>PUNTOS CRÍTICOS AGUDOS</b><br>(Alto Dinámico, Baja Estructural)",xanchor='right',yanchor='bottom',align='right',**anno_defaults)
            fig.add_annotation(x=0.02,y=0.98,text="<b>AMENAZAS LATENTES</b><br>(Bajo Dinámico, Alta Estructural)",xanchor='left',yanchor='top',align='left',**anno_defaults)
            fig.add_annotation(x=0.02,y=0.02,text="ZONAS ESTABLES",xanchor='left',yanchor='bottom',align='left',**anno_defaults)
            fig.update_layout(title_text="<b>Matriz de Riesgo Estratégico</b>",title_x=0.5,title_font=dict(size=20,family="Arial,sans-serif"),xaxis_title="Riesgo Dinámico (Eventos y Recencia) →",yaxis_title="Vulnerabilidad Estructural (Intrínseca) →",height=550,plot_bgcolor='white',paper_bgcolor='white',showlegend=False,xaxis=dict(gridcolor='#e5e5e5',zeroline=False,range=[0,kpi_df['Ensemble Risk Score'].max()*1.1]),yaxis=dict(gridcolor='#e5e5e5',zeroline=False,range=[0,kpi_df['GNN_Structural_Risk'].max()*1.1]),margin=dict(l=80,r=40,t=100,b=80))
            st.plotly_chart(fig,use_container_width=True)
        except Exception as e:
            logger.error(f"Error en cuadrante de vulnerabilidad: {e}",exc_info=True); st.warning("No se pudo mostrar la Matriz de Riesgo Estratégico.")
            
    def _plot_allocation_opportunity(self, kpi_df: pd.DataFrame, allocations: Dict[str, int]):
        st.markdown("**Análisis:** Esta matriz identifica las brechas de recursos más críticas. Zonas en el **cuadrante superior derecho (Déficit Urgente)** son de alto riesgo y con recursos insuficientes. Zonas en el **cuadrante superior izquierdo (Excedente Potencial)** son de menor riesgo pero podrían tener unidades para reasignar.")
        try:
            df = kpi_df[['Zone', 'Integrated_Risk_Score', 'Expected Incident Volume']].copy()
            df['allocated_units'] = df['Zone'].map(allocations).fillna(0)
            risk_cov = self.config.get('model_params', {}).get('risk_coverage_per_unit', CONSTANTS['RISK_COVERAGE_PER_UNIT'])
            df['risk_covered'] = df['allocated_units'] * risk_cov
            df['resource_deficit'] = df['Integrated_Risk_Score'] - df['risk_covered']
            max_incidents = max(df['Expected Incident Volume'].max(), 1); sizeref = 2. * max_incidents / (40.**2)
            mean_risk = df['Integrated_Risk_Score'].mean()
            fig = go.Figure()
            fig.add_shape(type="rect",xref="paper",yref="paper",x0=0.5,y0=0.5,x1=1,y1=1,line=dict(width=0),fillcolor="rgba(211,47,47,0.1)",layer="below")
            fig.add_shape(type="rect",xref="paper",yref="paper",x0=0,y0=0.5,x1=0.5,y1=1,line=dict(width=0),fillcolor="rgba(30,136,229,0.1)",layer="below")
            fig.add_trace(go.Scatter(x=df['Integrated_Risk_Score'],y=df['resource_deficit'],mode='markers',marker=dict(size=df['Expected Incident Volume'],sizemode='area',sizeref=sizeref,sizemin=4,color=df['resource_deficit'],colorscale="OrRd",showscale=True,colorbar=dict(title="Déficit<br>Recursos",x=1.15)),customdata=df[['Zone','allocated_units']],hovertemplate="<b>Zona: %{customdata[0]}</b><br>Riesgo Total: %{x:.3f}<br>Déficit de Recursos: %{y:.3f}<br>Unidades Asignadas: %{customdata[1]}<extra></extra>"))
            df_high_priority = df[df['resource_deficit'] > 0.1]
            fig.add_trace(go.Scatter(x=df_high_priority['Integrated_Risk_Score'],y=df_high_priority['resource_deficit'],mode='text',text=df_high_priority['Zone'],textposition="top center",textfont=dict(size=10,color='#444'),showlegend=False,hoverinfo='none'))
            fig.add_vline(x=mean_risk,line_width=1,line_dash="dash",line_color="darkgrey"); fig.add_hline(y=0,line_width=2,line_color="black")
            anno_font = dict(family="Arial,sans-serif",size=12,color="white")
            fig.add_annotation(xref="paper",yref="paper",x=0.98,y=0.98,text="<b>DÉFICIT URGENTE</b><br>(Alto Riesgo, Alta Necesidad)",showarrow=False,font=anno_font,bgcolor="#D32F2F",xanchor='right',yanchor='top',borderpad=4,bordercolor="#D32F2F")
            fig.add_annotation(xref="paper",yref="paper",x=0.02,y=0.98,text="<b>EXCEDENTE POTENCIAL</b><br>(Bajo Riesgo, Alta Necesidad)",showarrow=False,font=anno_font,bgcolor="#1E90FF",xanchor='left',yanchor='top',borderpad=4,bordercolor="#1E90FF")
            fig.add_annotation(xref="paper",yref="paper",x=0.98,y=0.02,text="<b>ESTABLE</b> (Alto Riesgo, Cubierto)",showarrow=False,font=dict(family="Arial,sans-serif",size=12,color="#333"),xanchor='right',yanchor='bottom',borderpad=4)
            fig.add_annotation(xref="paper",yref="paper",x=0.02,y=0.02,text="<b>ADECUADO</b> (Bajo Riesgo, Cubierto)",showarrow=False,font=dict(family="Arial,sans-serif",size=12,color="#333"),xanchor='left',yanchor='bottom',borderpad=4)
            fig.update_layout(title_text="Matriz de Oportunidad de Asignación",xaxis_title="Perfil de Riesgo de la Zona →",yaxis_title="← Necesidad de Recursos (Déficit) →",height=550,plot_bgcolor='white',showlegend=False,xaxis=dict(gridcolor='#e5e5e5',zeroline=False),yaxis=dict(gridcolor='#e5e5e5',zeroline=True,zerolinewidth=2,zerolinecolor='black'),margin=dict(l=60,r=40,t=60,b=60))
            st.plotly_chart(fig,use_container_width=True)
        except Exception as e:
            logger.error(f"Error en gráfico de oportunidad de asignación: {e}",exc_info=True); st.warning("No se pudo mostrar el gráfico de Oportunidad de Asignación.")
# --- START: INSERT THE NEW METHOD HERE (around line 543) ---

    def _plot_risk_momentum(self, kpi_df: pd.DataFrame):
        """
        [SME VISUALIZATION] Plots the change in risk from the previous period to the current one.
        This "momentum" chart quickly identifies zones that are heating up or cooling down.
        """
        st.markdown("**Análisis:** Este gráfico muestra el 'momentum' del riesgo—qué tan rápido está cambiando el riesgo para cada zona. Las barras rojas indican zonas donde el riesgo está aumentando ('calentándose'), mientras que las barras verdes muestran zonas donde el riesgo está disminuyendo ('enfriándose'). Esto es crucial para la asignación proactiva de recursos.")
        try:
            # We need at least one historical data point to calculate momentum.
            if 'historical_data' not in st.session_state or not st.session_state.historical_data:
                st.info("No hay datos históricos disponibles para calcular la tendencia del riesgo. Por favor, cargue un archivo de historial.")
                return

            # To get a 'previous' state, we run the KPI generation on the last historical snapshot.
            # We must use the core `generate_kpis` method, not the one with sparklines.
            last_historical_point = st.session_state.historical_data[-1]
            historical_incidents = last_historical_point.get('incidents', [])
            
            # Note: A more advanced implementation might also retrieve historical EnvFactors.
            # For this visualization, using current factors is a reasonable proxy for context.
            prev_kpi_df = self.engine.generate_kpis(
                st.session_state.historical_data[:-1], # History before the last point
                st.session_state.env_factors,          # Current environmental context
                historical_incidents                   # Incidents from the last historical point
            )

            if prev_kpi_df.empty:
                st.info("No se pudo calcular el estado de riesgo anterior a partir de los datos históricos.")
                return

            # Prepare data for comparison
            df = kpi_df[['Zone', 'Integrated_Risk_Score']].copy()
            prev_risk = prev_kpi_df.set_index('Zone')['Integrated_Risk_Score']
            df = df.join(prev_risk.rename('Prev_Risk_Score'), on='Zone')
            df.fillna(0, inplace=True) # Fill zones that might not have existed before

            # Calculate momentum and assign color
            df['momentum'] = df['Integrated_Risk_Score'] - df['Prev_Risk_Score']
            df['color'] = df['momentum'].apply(lambda m: '#D32F2F' if m > 0.001 else '#388E3C')
            df = df.sort_values('momentum', ascending=True)

            # Create the plot
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df['momentum'],
                y=df['Zone'],
                orientation='h',
                marker_color=df['color'],
                hovertemplate="<b>Zona: %{y}</b><br>Cambio en Riesgo: %{x:+.3f}<extra></extra>"
            ))

            fig.add_vline(x=0, line_width=2, line_color='black')

            fig.update_layout(
                title_text="<b>Tendencia del Riesgo (Momentum) por Zona</b>",
                xaxis_title="Cambio en el Puntaje de Riesgo Integrado (Actual vs. Anterior)",
                yaxis_title=None,
                height=500,
                plot_bgcolor='white',
                showlegend=False,
                xaxis=dict(gridcolor='#e5e5e5'),
                yaxis=dict(showgrid=False),
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            logger.error(f"Error al graficar el momentum del riesgo: {e}", exc_info=True)
            st.warning("No se pudo mostrar el gráfico de Tendencia del Riesgo.")

# --- END: NEW METHOD INSERTED ---
            
    def _plot_critical_zone_anatomy(self, kpi_df: pd.DataFrame):
        st.markdown("**Analysis:** This chart dissects the *composition* of risk for the most critical zones. Each horizontal line represents a zone, ordered by total risk. The **icon and its position** show the magnitude of each primary risk driver, allowing for direct comparison. The **total integrated risk score** is annotated on the right for context.")
        try:
            st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">', unsafe_allow_html=True)
            risk_cols = ['Violence Clustering Score', 'Accident Clustering Score', 'Medical Surge Score']
            if not all(col in kpi_df.columns for col in risk_cols): st.error("Data missing for Zone Anatomy plot."); return
            df_top = kpi_df.nlargest(7, 'Integrated_Risk_Score').copy()
            df_top = df_top.sort_values('Integrated_Risk_Score', ascending=True)
            fig = go.Figure()
            risk_visuals = {'Violence Clustering Score': {'name': 'Violence', 'color': '#D32F2F', 'icon_html': '<span style="font-family: \'Font Awesome 5 Free\'; font-weight: 900; color: white; font-size: 14px;"></span>'}, 'Accident Clustering Score': {'name': 'Accident', 'color': '#FBC02D', 'icon_html': '<span style="font-family: \'Font Awesome 5 Free\'; font-weight: 900; color: white; font-size: 14px;"></span>'}, 'Medical Surge Score': {'name': 'Medical',  'color': '#1E90FF', 'icon_html': '<span style="font-family: \'Font Awesome 5 Free\'; font-weight: 900; color: white; font-size: 14px;"></span>'}}
            for i, row in df_top.iterrows():
                risk_values = sorted([row[col] for col in risk_cols])
                fig.add_shape(type='line', x0=risk_values[0], y0=row['Zone'], x1=risk_values[-1], y1=row['Zone'], line=dict(color='rgba(0,0,0,0.2)', width=2), layer='below')
            for col, visual_props in risk_visuals.items():
                fig.add_trace(go.Scatter(x=df_top[col], y=df_top['Zone'], mode='markers+text', name=visual_props['name'], marker=dict(color=visual_props['color'], size=28, symbol='circle'), text=[visual_props['icon_html']] * len(df_top), textfont=dict(size=14), hovertemplate="<b>Zone:</b> %{y}<br><b>Risk Type:</b> %{name}<br><b>Score:</b> %{x:.3f}<extra></extra>"))
            for i, row in df_top.iterrows():
                fig.add_annotation(xref='paper', yref='y', x=1.01, y=row['Zone'], text=f"<b>{row['Integrated_Risk_Score']:.2f}</b>", showarrow=False, font=dict(size=14, color='#37474F', family="Arial Black, sans-serif"), align="left")
            fig.update_layout(title_text="<b>Risk Anatomy of Critical Zones</b>", title_x=0.5, xaxis_title="Component Risk Score", yaxis_title=None, height=500, plot_bgcolor='white', paper_bgcolor='white', showlegend=True, legend_title_text='Risk Driver', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font_size=12, itemsizing='constant'), xaxis=dict(showgrid=True, gridcolor='rgba(221, 221, 221, 0.7)', zeroline=False, range=[0, max(0.6, df_top[risk_cols].max().max() * 1.1)]), yaxis=dict(showgrid=False), margin=dict(t=80, b=40, l=40, r=60))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error in critical zone anatomy plot: {e}", exc_info=True); st.warning("Could not display Critical Zone Anatomy plot.")
            
    def _plot_critical_zone_anatomy(self, kpi_df: pd.DataFrame):
        """
        [SME VISUALIZATION] Plots a detailed Risk Composition Profile using an Icon-Enhanced Dot Plot.
        This elegant design allows for instant, intuitive comparison of risk components across zones.
        """
        st.markdown("**Análisis:** Este gráfico disecciona la *composición* del riesgo para las zonas más críticas. Cada línea horizontal representa una zona, ordenada por riesgo total. El **ícono y su posición** muestran la magnitud de cada impulsor de riesgo primario, permitiendo una comparación directa entre zonas. El **puntaje total de riesgo** se anota a la derecha para dar un contexto de la magnitud general.")
        try:
            # Inject Font Awesome CSS for icon rendering. This is robust and works in most environments.
            st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">', unsafe_allow_html=True)
            
            risk_cols = ['Violence Clustering Score', 'Accident Clustering Score', 'Medical Surge Score']
            if not all(col in kpi_df.columns for col in risk_cols):
                st.error("Faltan datos para el gráfico de Anatomía de Zona."); return

            df_top = kpi_df.nlargest(7, 'Integrated_Risk_Score').copy()
            df_top = df_top.sort_values('Integrated_Risk_Score', ascending=True) # Sort for bottom-to-top display

            fig = go.Figure()
            
            # --- Define the visual grammar with CORRECT professional Font Awesome icons ---
            risk_visuals = {
                'Violence Clustering Score': {'name': 'Violencia', 'color': '#D32F2F', 'icon_html': '<span style="font-family: \'Font Awesome 5 Free\'; font-weight: 900; color: white; font-size: 14px;"></span>'}, # fa-fire
                'Accident Clustering Score': {'name': 'Accidente', 'color': '#FBC02D', 'icon_html': '<span style="font-family: \'Font Awesome 5 Free\'; font-weight: 900; color: white; font-size: 14px;"></span>'}, # fa-car-crash
                'Medical Surge Score':       {'name': 'Médico',  'color': '#1E90FF', 'icon_html': '<span style="font-family: \'Font Awesome 5 Free\'; font-weight: 900; color: white; font-size: 14px;"></span>'}  # fa-heartbeat
            }

            # Add the connecting lines for each zone (the "dumbbell" bar)
            for i, row in df_top.iterrows():
                risk_values = sorted([row[col] for col in risk_cols])
                fig.add_shape(
                    type='line',
                    x0=risk_values[0], y0=row['Zone'],
                    x1=risk_values[-1], y1=row['Zone'],
                    line=dict(color='rgba(0,0,0,0.2)', width=2),
                    layer='below'
                )

            # Add the dots and icons as text markers
            for col, visual_props in risk_visuals.items():
                fig.add_trace(go.Scatter(
                    x=df_top[col],
                    y=df_top['Zone'],
                    mode='markers+text',
                    name=visual_props['name'],
                    marker=dict(
                        color=visual_props['color'],
                        size=28,
                        symbol='circle'
                    ),
                    text=[visual_props['icon_html']] * len(df_top),
                    textfont=dict(size=14), # Size is now controlled within the HTML span
                    hovertemplate="<b>Zona:</b> %{y}<br><b>Tipo de Riesgo:</b> %{name}<br><b>Puntaje:</b> %{x:.3f}<extra></extra>"
                ))
            
            # Add annotations for the total Integrated Risk Score on the right
            for i, row in df_top.iterrows():
                fig.add_annotation(
                    xref='paper', yref='y',
                    x=1.01, y=row['Zone'],
                    text=f"<b>{row['Integrated_Risk_Score']:.2f}</b>",
                    showarrow=False,
                    font=dict(size=14, color='#37474F', family="Arial Black, sans-serif"),
                    align="left"
                )

            fig.update_layout(
                title_text="<b>Anatomía del Riesgo en Zonas Críticas</b>",
                title_x=0.5,
                xaxis_title="Puntaje del Componente de Riesgo",
                yaxis_title=None,
                height=500,
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=True,
                legend_title_text='Impulsor de Riesgo',
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font_size=12,
                    itemsizing='constant' # Ensures legend markers are uniform
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(221, 221, 221, 0.7)',
                    zeroline=False,
                    range=[0, max(0.6, df_top[risk_cols].max().max() * 1.15)] # Extend range for annotations
                ),
                yaxis=dict(showgrid=False),
                margin=dict(t=80, b=40, l=40, r=60) # Add right margin for total score annotations
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error en gráfico de anatomía de zona crítica: {e}", exc_info=True)
            st.warning("No se pudo mostrar el gráfico de Anatomía de Zona Crítica.")
            
    def _plot_risk_contribution_sunburst(self, kpi_df: pd.DataFrame):
        """
        [SME VISUALIZATION] Plots a Risk Diagnostics sunburst chart to dissect the nature of a zone's risk.
        """
        st.markdown("""
        **Análisis:** Este gráfico de diagnóstico desglosa el `Puntaje de Riesgo Integrado` para entender su naturaleza. Permite responder a la pregunta: *"¿Este riesgo es predecible y se basa en patrones históricos, o es una amenaza compleja y novedosa detectada por la IA?"*

        **Cómo Interpretar:**
        - **Anillo Azul (Base):** Representa el riesgo convencional basado en datos históricos y estadísticos. Un segmento azul grande significa que el riesgo es **predecible**.
        - **Anillo Rojo/Naranja (Avanzado):** Representa riesgos complejos detectados por modelos de IA (proximidad, tensión del sistema, etc.). Un segmento rojo grande significa que la amenaza es **novedosa o atípica** y requiere una respuesta más matizada.
        """)
        try:
            zones = kpi_df.nlargest(5, 'Integrated_Risk_Score')['Zone'].tolist()
            if not zones:
                st.info("No hay zonas de alto riesgo para analizar."); return
                
            zone = st.selectbox("Seleccione una Zona de Alto Riesgo para Diagnóstico:", options=zones, key="sunburst_zone_select")
            if not zone: return

            z_data = kpi_df.loc[kpi_df['Zone'] == zone].iloc[0]
            weights = self.config.get('model_params', {}).get('advanced_model_weights', {})
            
            total_risk = z_data.get('Integrated_Risk_Score', 0)
            if total_risk < 1e-9: # Avoid division by zero
                st.info(f"La zona {zone} no presenta un riesgo significativo para desglosar.")
                return

            base_value = weights.get('base_ensemble', 0) * z_data.get('Ensemble Risk Score', 0)
            stgp_value = weights.get('stgp', 0) * z_data.get('STGP_Risk', 0)
            hmm_value = weights.get('hmm', 0) * z_data.get('HMM_State_Risk', 0)
            gnn_value = weights.get('gnn', 0) * z_data.get('GNN_Structural_Risk', 0)
            gt_value = weights.get('game_theory', 0) * z_data.get('Game_Theory_Tension', 0)
            advanced_value = stgp_value + hmm_value + gnn_value + gt_value

            data = {
                'ids': ['Total', 'Base Ensemble', 'Advanced Models', 'STGP Risk', 'HMM State', 'GNN Structure', 'Game Tension'],
                'labels': ['Puntaje Total', 'Riesgo Base', 'Riesgo Avanzado', 'Proximidad', 'Patrón Anómalo', 'Estructural', 'Tensión Sistémica'],
                'parents': ['', 'Total', 'Total', 'Advanced Models', 'Advanced Models', 'Advanced Models', 'Advanced Models'],
                'values': [total_risk, base_value, advanced_value, stgp_value, hmm_value, gnn_value, gt_value],
                'customdata': [100, (base_value/total_risk)*100, (advanced_value/total_risk)*100, (stgp_value/total_risk)*100, (hmm_value/total_risk)*100, (gnn_value/total_risk)*100, (gt_value/total_risk)*100],
                'marker': {
                    'colors': [
                        '#E0E0E0',  # Total (neutral)
                        '#1E90FF',  # Base Ensemble (Predictable Risk)
                        '#D32F2F',  # Advanced Models (Complex Risk)
                        '#FF6347',  # STGP
                        '#FF7F50',  # HMM
                        '#FF8C00',  # GNN
                        '#FFA500'   # Game Theory
                    ]
                }
            }

            fig = go.Figure(go.Sunburst(
                ids=data['ids'], 
                labels=data['labels'], 
                parents=data['parents'], 
                values=data['values'],
                branchvalues="total",
                marker=data['marker'],
                customdata=data['customdata'],
                hovertemplate='<b>%{label}</b><br>Contribución al Riesgo: %{value:.3f}<br>Porcentaje del Total: %{customdata:.1f}%<extra></extra>',
                insidetextorientation='radial'
            ))
            
            fig.update_layout(
                margin=dict(t=20, l=0, r=0, b=0),
                title_text=f"Diagnóstico de Riesgo para la Zona: {zone}",
                title_x=0.5,
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error en gráfico sunburst de contribución: {e}", exc_info=True)
            st.warning("No se pudo mostrar el gráfico de Contribución de Riesgo.")
            
    def _plot_forecast_with_uncertainty(self):
        """
        [SME VISUALIZATION] Plots the 72-hour risk forecast with enhanced context and interactivity.
        This includes confidence intervals and a critical risk threshold.
        """
        st.markdown("""
        **Análisis:** Este gráfico proyecta la trayectoria del riesgo para las zonas seleccionadas durante las próximas 72 horas.
        **Cómo Interpretar:**
        - **Línea Sólida:** La predicción más probable del puntaje de riesgo.
        - **Área Sombreada:** El intervalo de confianza del 95%. Un área más ancha indica una mayor incertidumbre en la predicción.
        - **Línea Roja Punteada:** El umbral de riesgo crítico. Permite identificar rápidamente cuándo se espera que una zona entre en un estado de alta peligrosidad.
        """)
        try:
            fc_df, kpi_df = st.session_state.forecast_df, st.session_state.kpi_df
            if fc_df.empty or kpi_df.empty:
                st.info("No hay datos de pronóstico para mostrar."); return
            
            zones = sorted(fc_df['Zone'].unique().tolist())
            defaults = kpi_df.nlargest(3, 'Integrated_Risk_Score')['Zone'].tolist()
            
            selected_zones = st.multiselect("Seleccione Zonas para Pronóstico:", options=zones, default=defaults, key="forecast_zone_select")
            
            if not selected_zones:
                st.info("Por favor, seleccione al menos una zona para visualizar el pronóstico."); return

            fig = go.Figure()
            colors = px.colors.qualitative.Plotly

            for i, zone in enumerate(selected_zones):
                zone_df = fc_df[fc_df['Zone'] == zone]
                if zone_df.empty: continue
                
                color = colors[i % len(colors)]
                rgb = px.colors.hex_to_rgb(color)
                
                # Add uncertainty band (area)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([zone_df['Horizon (Hours)'], zone_df['Horizon (Hours)'][::-1]]),
                    y=np.concatenate([zone_df['Upper_Bound'], zone_df['Lower_Bound'][::-1]]),
                    fill='toself',
                    fillcolor=f'rgba({",".join(map(str,rgb))}, 0.15)',
                    line={'color': 'rgba(255,255,255,0)'},
                    hoverinfo="skip",
                    showlegend=False
                ))
                
                # Add main forecast line
                fig.add_trace(go.Scatter(
                    x=zone_df['Horizon (Hours)'],
                    y=zone_df['Combined Risk'],
                    name=zone,
                    line=dict(color=color, width=2.5),
                    mode='lines+markers',
                    hovertemplate=f"<b>{zone}</b><br>Hora: %{{x}}h<br>Riesgo Proyectado: %{{y:.3f}}<extra></extra>"
                ))
            
            # --- Add Critical Risk Threshold for context ---
            critical_threshold = 0.75  # Example threshold
            fig.add_hline(
                y=critical_threshold,
                line_dash="dash",
                line_color="#D32F2F",
                line_width=2,
                annotation_text="Umbral de Riesgo Crítico",
                annotation_position="bottom right",
                annotation_font=dict(color="#D32F2F", size=12)
            )

            fig.update_layout(
                title_text="<b>Pronóstico de Trayectoria de Riesgo y Incertidumbre</b>",
                title_x=0.5,
                xaxis_title="Horizonte de Pronóstico (Horas)",
                yaxis_title="Puntaje de Riesgo Integrado Proyectado",
                height=500,
                plot_bgcolor='white',
                paper_bgcolor='white',
                legend_title_text='Zona',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
                xaxis=dict(gridcolor='#e5e5e5'),
                yaxis=dict(gridcolor='#e5e5e5', range=[0, 1])
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error en gráfico de pronóstico: {e}", exc_info=True)
            st.warning("No se pudo mostrar el gráfico de pronóstico.")
    # --- Metodología, etc. ---
    def _render_methodology_tab(self):
        st.header("Arquitectura y Metodología del Sistema")
        st.markdown("Esta sección ofrece una inmersión profunda en el motor analítico que impulsa la plataforma Phoenix v4.0. Está diseñada para científicos de datos, analistas y personal de mando que deseen comprender el 'porqué' detrás de las predicciones y prescripciones del sistema.")
        self._render_architecture_philosophy()
        self._render_prediction_engine()
        self._render_prescription_engine()
        self._render_incident_specific_weighting()
        self._render_kpi_glossary()

    def _render_architecture_philosophy(self):
        with st.expander("I. Filosofía Arquitectónica: De la Predicción a la Prescripción", expanded=True):
            st.markdown("""
            El objetivo fundamental de RedShield AI: Phoenix v4.0 es diseñar un cambio de paradigma en la respuesta a emergencias: de un **modelo reactivo** tradicional (despachar unidades después de que ocurra un incidente) a una **postura proactiva y prescriptiva** (anticipar dónde es probable que surjan los incidentes y posicionar recursos de forma prescriptiva para minimizar los tiempos de respuesta y maximizar el impacto).
            
            Para lograr esto, el sistema se basa en una filosofía de **Modelado Jerárquico de Ensamble**. En lugar de depender de un único algoritmo de "caja negra", Phoenix v4.0 integra una cartera diversa de técnicas analíticas en una arquitectura de "caja de cristal" por capas. Esto crea un sistema altamente robusto y resiliente donde las debilidades de un modelo se compensan con las fortalezas de otros.
            
            La arquitectura se compone de cuatro capas principales:
            1. **CAPA 1: Modelos Fundacionales.** Consiste en modelos estadísticos bien establecidos (Procesos de Hawkes, Redes Bayesianas, Laplacianos de Grafos) que crean una comprensión base robusta del riesgo. Esto produce el `Puntaje_Riesgo_Ensamble`.
            2. **CAPA 2: IA Avanzada y Proxies de Complejidad.** Introduce "proxies" (representaciones) computacionalmente económicos pero potentes para modelos de vanguardia (ST-GPs, HMMs, GNNs) para capturar patrones más profundos y matizados que complementan la capa fundacional.
            3. **CAPA 3: Síntesis Integrada (Predicción).** Las salidas de las dos primeras capas se combinan en una síntesis final ponderada para producir el `Puntaje_Riesgo_Integrado` definitivo. Este puntaje representa la mejor **predicción** de riesgo del sistema.
            4. **CAPA 4: Optimización Prescriptiva (Prescripción).** El `Puntaje_Riesgo_Integrado` y el `Volumen_Incidentes_Esperado` se introducen en un motor de Investigación de Operaciones (IO). Esta capa va más allá de la predicción para llegar a la **prescripción**, determinando la acción *óptima* del mundo real a tomar (p. ej., asignación de ambulancias) para mitigar mejor el riesgo predicho.
            """)

    def _render_prediction_engine(self):
        with st.expander("II. El Motor de Predicción: Un Análisis Profundo Multi-modelo", expanded=False):
            st.info("#### Principio Fundamental: Los incidentes urbanos de emergencia presentan una naturaleza compleja y dinámica, lo que imposibilita la aplicación de un modelo único capaz de abordar la totalidad de escenarios posibles. En respuesta a esta limitación, RedShield AI emplea herramientas avanzadas provenientes de diversas ramas de las matemáticas, seleccionadas en función del tipo y del contexto del análisis requerido: diagnóstico, predictivo o prescriptivo. Los algoritmos implementados por RedShield AI se basan en principios de aprendizaje continuo, alimentándose de datos actualizados en tiempo real de forma fluida y dinámica. Su enfoque metodológico integra de manera flexible modelos matemáticos, estadísticos, computacionales y cualitativos propios de la inteligencia artificial, lo que permite una adaptación efectiva al contexto del incidente y respalda la toma de decisiones informadas, oportunas y basadas en evidencia.", icon="💡")
            st.markdown("---")
            st.markdown("#### **A. Modelos Estocásticos y Estadísticos (El 'Cuándo')**")
            st.markdown("""
            * **Proceso de Poisson No Homogéneo (NHPP):** Este modelo forma la columna vertebral temporal de nuestras predicciones. Entiende que las tasas de incidentes no son constantes.
                - **Pregunta que Responde:** *"¿Cuál es la probabilidad base de un incidente a las 3 AM de un martes frente a las 6 PM de un viernes?"*
                - **Relevancia:** Captura la naturaleza predecible y cíclica de la vida urbana, asegurando que nuestro riesgo base sea sensible a la hora del día y al día de la semana.
                - **Formulación Matemática:** La función de intensidad `λ(t)` se modela como una función del tiempo, a menudo utilizando un modelo log-lineal con términos de Fourier para capturar la ciclicidad:
            """)
            st.latex(r'''\lambda(t) = \exp\left( \beta_0 + \sum_{k=1}^{K} \left[ \beta_k \cos\left(\frac{2\pi kt}{T}\right) + \gamma_k \sin\left(\frac{2\pi kt}{T}\right) \right] \right)''')
            st.markdown("""
                    donde `β` y `γ` son coeficientes aprendidos de datos históricos y `T` es el período del ciclo (p. ej., 24 horas o 168 horas).
            * **Proceso de Hawkes (Proceso de Puntos Auto-excitado):** Es la piedra angular de nuestros modelos de violencia y accidentes en cascada. Opera bajo el principio de que ciertos eventos pueden desencadenar "réplicas".
                - **Pregunta que Responde:** *"Dado que acaba de ocurrir un tiroteo, ¿cuál es el riesgo inmediato y elevado de otro tiroteo en la misma área?"* o *"Tras una colisión grave en una autopista, ¿cuál es la probabilidad aumentada de accidentes secundarios debido a la congestión del tráfico?"*
                - **Relevancia:** Crítico para modelar la violencia de pandillas por represalias y los incidentes de tráfico en cadena. Impulsa directamente el `Puntaje_Agrupamiento_Trauma`.
                - **Formulación Matemática:** La intensidad condicional `λ(t)` de un evento en el tiempo `t` se define como:
            """)
            st.latex(r'''\lambda(t) = \mu(t) + \sum_{t_i < t} \alpha \cdot g(t - t_i)''')
            st.markdown("""
                    donde `μ(t)` es la tasa de fondo del NHPP, la suma es sobre los tiempos de eventos pasados `tᵢ`, `α` es la razón de ramificación (fuerza de la réplica), y `g(t - tᵢ)` es el núcleo de disparo que modela la influencia decreciente de los eventos pasados.
            * **Redes Bayesianas:** Estos modelos representan nuestra comprensión de las relaciones causales. Combinan tasas base estáticas con factores ambientales en tiempo real.
                - **Pregunta que Responde:** *"¿Cómo influyen colectivamente en la probabilidad de un incidente un día feriado, combinado con un clima lluvioso y un concierto importante?"*
                - **Relevancia:** Permite que el sistema razone con conocimiento experto y se adapte a factores contextuales como `Clima`, `Es Feriado`, y `Evento Principal`. Es un impulsor central de la `Probabilidad_Incidente` base.
                - **Formulación Matemática:** Basado en la regla de la cadena de la probabilidad, donde la probabilidad conjunta es el producto de las probabilidades condicionales: $P(X_1, ..., X_n) = \\prod_{i=1}^{n} P(X_i | \\text{Padres}(X_i))$. Nuestra red modela `P(TasaIncidentes | Clima, Feriado, ...)` para encontrar la tasa base más probable.
            """)
            st.markdown("---")
            st.markdown("#### **B. Modelos Espaciotemporales y de Grafos (El 'Dónde' y 'Cómo se Propaga')**")
            st.markdown("""
            * **Procesos Gaussianos Espaciotemporales (ST-GPs):** Nuestro KPI `Riesgo_STGP` es un proxy de esta técnica avanzada. Modela el riesgo como un fluido continuo sobre el mapa.
                - **Pregunta que Responde:** *"Un incidente ocurrió a 500 metros del límite de esta zona. ¿Cuánta 'presión de riesgo' ejerce eso sobre esta zona?"*
                - **Relevancia:** Interpola el riesgo de manera inteligente en el mapa, asegurando que la proximidad al peligro siempre se tenga en cuenta, incluso a través de límites de zona arbitrarios.
                - **Formulación Matemática:** El riesgo `f` en un punto espaciotemporal `(s, t)` se modela como una muestra de un Proceso Gaussiano:
            """)
            st.latex(r'''f(s, t) \sim \mathcal{GP}(m(s, t), k((s, t), (s', t')))''')
            st.markdown("""
                    donde `m(s, t)` es la función de media y `k` es un núcleo espaciotemporal, a menudo un producto de un núcleo espacial (como RBF) y uno temporal:
            """)
            st.latex(r'''k((s, t), (s', t')) = \sigma^2 \exp\left(-\frac{\|s-s'\|^2}{2l_s^2}\right) \exp\left(-\frac{|t-t'|^2}{2l_t^2}\right)''')
            st.markdown("""
                    Las escalas de longitud `l_s` y `l_t` controlan el "rango" de influencia en el espacio y el tiempo.
            * **Redes Neuronales de Grafos (GNNs):** La red de carreteras de la ciudad y las adyacencias de las zonas se tratan como un grafo complejo. Una GNN aprende una comprensión profunda y estructural del rol de cada zona dentro de esta red.
                - **Pregunta que Responde:** *"¿Es esta zona inherentemente vulnerable simplemente por su posición como un cruce importante, independientemente de los eventos recientes?"*
                - **Relevancia:** Identifica vulnerabilidades estructurales a largo plazo que pueden no ser evidentes solo a partir de datos de incidentes recientes. Impulsa el `Riesgo_Estructural_GNN`, que representa el riesgo intrínseco de una zona.
                - **Formulación Matemática (Capa GCN):** Una GNN funciona pasando "mensajes" entre nodos conectados. Una operación de capa común es:
            """)
            st.latex(r'''H^{(l+1)} = \sigma(\hat{D}^{-\frac{1}{2}} \hat{A} \hat{D}^{-\frac{1}{2}} H^{(l)} W^{(l)})''')
            st.markdown("""
                    Esto significa que las nuevas características para un nodo (`H^(l+1)`) son un agregado transformado de las características anteriores de sus vecinos (`H^(l)`), donde `Â` es la matriz de adyacencia con auto-bucles, `D̂` es su matriz de grados, `W` es una matriz de pesos aprendible, y `σ` es una función de activación.
            * **Difusión Laplaciana en Grafos:** Esta técnica modela cómo los efectos (como el tráfico, el pánico o los cordones policiales) se "desbordan" de una zona a sus vecinas a través de la red de carreteras.
                - **Pregunta que Responde:** *"Un gran incendio ha cerrado tres cuadras en la Zona A. ¿Cómo aumenta esto el riesgo de accidentes relacionados con el tráfico en la Zona B adyacente?"*
                - **Relevancia:** Esencial para modelar los efectos secundarios de incidentes mayores. Calcula directamente el `Riesgo_Desbordamiento_Espacial`.
                - **Formulación Matemática:** El proceso se modela mediante la ecuación de difusión de calor en un grafo. Un solo paso discreto de este proceso es:
            """)
            st.latex(r'''r(t+1) = (I - \epsilon L) r(t)''')
            st.markdown("""
                    donde `r(t)` es el vector de riesgos de las zonas, `I` es la matriz identidad, `ε` es un tamaño de paso pequeño, y `L` es la matriz laplaciana normalizada del grafo.
            """)
            st.markdown("---")
            st.markdown("#### **C. Complejidad y Teoría de la Información (El 'Estado del Sistema')**")
            st.markdown("""
            * **Exponentes de Lyapunov (Puntaje de Sensibilidad al Caos):** Un concepto de la Teoría del Caos que mide la sensibilidad de un sistema a pequeños cambios. Un puntaje alto significa que el sistema está en un estado frágil e impredecible.
                - **Pregunta que Responde:** *"¿Está la ciudad operando normalmente, o está en un estado 'frágil' donde un pequeño incidente podría convertirse en una crisis mayor?"*
                - **Relevancia:** Es una "alarma de inestabilidad" crítica para el personal de mando. No predice un incidente específico, pero advierte que todo el sistema es volátil.
                - **Formulación Matemática (Conceptual):** Mide la tasa exponencial de divergencia de trayectorias cercanas. Si `δ(t)` es la separación entre dos trayectorias a lo largo del tiempo, el mayor exponente de Lyapunov `λ` se estima mediante:
            """)
            st.latex(r'''\lambda \approx \frac{1}{t} \ln \frac{\|\delta(t)\|}{\|\delta(0)\|}''')
            st.markdown("""
                    Un `λ` positivo es un indicador de caos.
            * **Divergencia de Kullback-Leibler (KL) (Puntaje de Anomalía):** Una métrica de la teoría de la información que mide cuánto se desvía el patrón actual de incidentes de la norma histórica.
                - **Pregunta que Responde:** *"¿Estamos viendo la cantidad correcta de incidentes, pero en todos los lugares equivocados hoy? ¿O estamos viendo un nuevo tipo de incidente extraño que nunca antes habíamos visto?"*
                - **Relevancia:** Detecta "anomalías de patrón" que las métricas simples basadas en volumen pasarían por alto. Un puntaje alto es una señal clara de que "hoy no es un día normal".
                - **Formulación Matemática:**
            """)
            st.latex(r'''D_{KL}(P || Q) = \sum_{z \in \text{Zonas}} P(z) \log{\frac{P(z)}{Q(z)}}''')
            
    def _render_prescription_engine(self):
        with st.expander("III. El Motor de Prescripción: Asignación Óptima de Recursos", expanded=False):
            st.info("#### Principio Fundamental: Pasar de 'qué sucederá' a 'cuál es la mejor acción a tomar' en este momento y en un horizonte de futuro inmediato, basándose en datos actuales analizados en tiempo real.", icon="🎯")
            st.markdown("""
            El motor prescriptivo utiliza los puntajes de riesgo de la capa de predicción como entradas para modelos sofisticados de Investigación de Operaciones. Esto asegura que la asignación de recursos no sea solo intuitiva, sino matemáticamente óptima según nuestros objetivos.
            
            * **Programación Lineal Entera Mixta (MILP):** Es la herramienta principal para la asignación `Lineal Óptima`. Encuentra la forma probadamente mejor de asignar un número entero de ambulancias a las zonas.
                - **Objetivo:** Maximizar la cantidad total de riesgo "cubierto" en toda la ciudad.
                - **Relevancia:** Excelente para encontrar la solución más eficiente bajo un único objetivo claro. Es rápida y garantiza un resultado matemáticamente óptimo para un problema lineal.
                - **Formulación Matemática (Simplificada):**
            """)
            st.latex(r'''\begin{aligned} & \text{maximizar} && \sum_{i \in \text{Zonas}} R_i \cdot c_i \\ & \text{sujeto a} && \sum_{i \in \text{Zonas}} c_i \leq N, \quad c_i \in \mathbb{Z}^+ \end{aligned}''')
            st.markdown("""
                    donde `Rᵢ` es el puntaje de riesgo para la zona `i`, `cᵢ` es el número entero de ambulancias asignadas, y `N` es el total disponible.
            * **Programación No Lineal (NLP):** Es nuestro modelo más avanzado para la asignación `No Lineal Óptima`. Captura dinámicas complejas del mundo real que los modelos lineales omiten.
                - **Objetivo:** Minimizar una función de "insatisfacción del sistema", que incluye dos efectos no lineales clave:
                    1. **Rendimientos Decrecientes (Utilidad Logarítmica):** La primera ambulancia enviada a una zona proporciona un gran beneficio; la quinta, mucho menos. El modelo entiende esto y evita la sobresaturación de una única zona de alto riesgo si otra zona no tiene cobertura.
                    2. **Penalizaciones por Congestión (Penalización Cuadrática):** A medida que el número de incidentes esperados en una zona supera con creces las unidades asignadas, el "daño" (p. ej., el retraso en el tiempo de respuesta) crece exponencialmente, no linealmente.
                - **Relevancia:** Proporciona las recomendaciones más realistas y robustas. Realiza concesiones inteligentes que un humano o un modelo más simple podrían pasar por alto, lo que lleva a una postura general del sistema más resiliente.
                - **Formulación Matemática (Simplificada):**
            """)
            st.latex(r'''\begin{aligned} & \text{minimizar} && \sum_{i \in \text{Zonas}} \left( w_1(R_i - R_i \log(1+c_i)) + w_2 \left( \frac{E_i}{1+c_i} \right)^2 \right) \\ & \text{sujeto a} && \sum_{i \in \text{Zonas}} c_i = N, \quad c_i \geq 0 \end{aligned}''')
            st.markdown("""
                    donde `Eᵢ` son los incidentes esperados, el término `log` modela los **rendimientos decrecientes**, y el término cuadrático modela las **penalizaciones por congestión**.
            * **Teoría de Colas:** Esta teoría matemática se utiliza conceptualmente para modelar la tensión del sistema, particularmente en los hospitales.
                - **Relevancia:** Al comprender las tasas de llegada (de nuestras predicciones) y las tasas de servicio, podemos estimar mejor los tiempos de espera y el impacto de los desvíos hospitalarios, lo que alimenta el `Índice_Suficiencia_Recursos`.
                - **Objetivo:** Modelar y predecir matemáticamente los tiempos de espera, la congestión y la probabilidad de saturación del sistema (p. ej., en una sala de emergencias de un hospital).
                - **Relevancia General en el Sistema:** La teoría de colas proporciona una base teórica robusta para el `Índice_Suficiencia_Recursos` e informa las decisiones de enrutamiento. En lugar de una simple penalización por un hospital ocupado, permite al sistema calcular el *retraso esperado real*, lo que conduce a asignaciones más inteligentes.
                - **La Pregunta que Responde:** "Si enviamos otra ambulancia al Hospital X, ¿cuál es la probabilidad de que tenga que esperar más de 15 minutos para entregar al paciente, dada su carga actual de pacientes y nuestra tasa de llegada prevista de nuevos incidentes?"
                - **Formulación Matemática (Modelo M/M/c):** Para un sistema con `c` servidores (p. ej., camas de urgencias), una tasa de llegada de Poisson `λ`, y una tasa de servicio exponencial `μ` por servidor, la probabilidad de que un paciente que llega tenga que esperar se da por la **Fórmula de Erlang-C**.
            """)
            st.markdown(r'''$$ P_{\text{wait}} = C(c, \lambda/\mu) = \frac{\frac{(\lambda/\mu)^c}{c!}}{ \left(\sum_{k=0}^{c-1} \frac{(\lambda/\mu)^k}{k!}\right) + \frac{(\lambda/\mu)^c}{c! (1 - \frac{\lambda}{c\mu})}} $$''', unsafe_allow_html=True)
            st.markdown("""
                - **Relevancia Matemática:** Este es un pilar de la Investigación de Operaciones, que proporciona un marco analítico poderoso para comprender y optimizar sistemas estocásticos definidos por llegadas y tiempos de servicio aleatorios, lo que describe perfectamente una red de respuesta a emergencias.
            """)

    def _render_incident_specific_weighting(self):
        with st.expander("IV. Ponderación de Modelos Específica por Incidente", expanded=False):
            st.markdown("""
            El sistema no es de talla única. El `Puntaje_Riesgo_Integrado` final es una suma ponderada de muchas salidas de modelos, y estos pesos se ven influenciados dinámicamente por la naturaleza del riesgo que se está evaluando.
            
            #### **Trauma - Violencia**
            * **Modelos Predictivos Primarios:** Los **Procesos de Hawkes** son primordiales, ya que modelan explícitamente la naturaleza vengativa y auto-excitante de la violencia. El **Riesgo Estructural de GNN** también es crítico para identificar puntos calientes territoriales a largo plazo.
            * **Modelo Prescriptivo Primario:** A menudo se prefiere **NLP** no solo para cubrir el riesgo, sino también para evitar la sobresaturación de un área, lo que puede ser crucial en situaciones tácticas fluidas.
            
            #### **Trauma - Accidentes**
            * **Modelos Predictivos Primarios:** Las **Redes Bayesianas** (que incorporan el clima y el tráfico) y los **Laplacianos de Grafos** (que modelan el desbordamiento de la congestión) son los impulsores clave.
            * **Modelo Prescriptivo Primario:** **NLP** es muy eficaz aquí, ya que su penalización por congestión incorporada modela directamente la consecuencia real de los atascos de tráfico, lo que lleva a decisiones de posicionamiento más inteligentes.

            #### **Emergencias Médicas**
            * **Modelos Predictivos Primarios:** Las **Redes Bayesianas** son cruciales para incorporar factores ambientales como olas de calor y calidad del aire. Los modelos espaciotemporales que analizan la densidad de población y la demografía (p. ej., la edad) también son clave.
            * **Modelo Prescriptivo Primario:** La elección del modelo está fuertemente influenciada por el **Estado de Desvío de Hospitales**. Cuando los hospitales están bajo presión, los modelos prescriptivos deben sopesar no solo el riesgo de un incidente, sino también el tiempo de viaje adicional y el riesgo de retraso a la llegada, un factor que NLP puede incorporar de forma más natural.
            """)

    def _render_kpi_glossary(self):
        with st.expander("V. Glosario de Indicadores Clave de Desempeño (KPI)", expanded=False):
            kpi_defs = {
                "Puntaje de Riesgo Integrado": {"description": "La métrica de riesgo final y sintetizada, usada para todas las decisiones operativas. Es la única fuente de verdad para priorizar zonas y asignar recursos.", "question": "¿Qué zona necesita más mi atención ahora mismo, considerando todos los factores?", "relevance": "Impulsa las recomendaciones finales de asignación de recursos y proporciona el ranking primario de zonas por nivel de amenaza general.", "formula": r'''R_{\text{int}} = \sum_{i} w_i \cdot \text{ComponenteRiesgo}_i'''},
                "Puntaje de Riesgo de Ensamble": {"description": "Un puntaje combinado de modelos estadísticos fundacionales que representa el riesgo base y estable para una zona.", "question": "¿Cuál es el nivel 'normal' o esperado de riesgo para esta zona, dado el contexto actual (ej. clima, día)?", "relevance": "Proporciona una evaluación de riesgo robusta y menos volátil, previniendo reacciones exageradas a eventos menores y transitorios.", "formula": r'''R_{\text{ens}} = \sum_{k=1}^{K} w_k \cdot \text{normalizar}(M_k)'''},
                "Riesgo Estructural (GNN)": {"description": "Una medida de la vulnerabilidad intrínseca y a largo plazo de una zona, basada en su posición y conectividad dentro de las redes viales y sociales de la ciudad.", "question": "¿Es esta zona inherentemente peligrosa, independientemente de eventos recientes?", "relevance": "Identifica áreas crónicamente en riesgo que pueden requerir intervención estratégica a largo plazo (ej. vigilancia comunitaria, cambios de infraestructura).", "formula": r'''\text{PageRank}(z_i) = \frac{1-d}{N} + d \sum_{z_j \in N(z_i)} \frac{\text{PR}(z_j)}{|N(z_j)|}'''},
                "Riesgo Espaciotemporal (STGP)": {"description": "Un puntaje que representa el riesgo que irradia de incidentes graves recientes, decayendo sobre el espacio y el tiempo.", "question": "¿Cuánta 'presión de riesgo' ejerce un incidente importante en una zona vecina sobre esta?", "relevance": "Captura la correlación espaciotemporal del riesgo, asegurando que la proximidad al peligro se tenga en cuenta.", "formula": r'''f(s, t) \sim \mathcal{GP}(m(s, t), k((s, t), (s', t')))'''},
                "Tensión (Teoría de Juegos)": {"description": "Una métrica que cuantifica la contribución de una zona a la competencia por recursos a nivel de sistema.", "question": "¿Qué zonas están causando la mayor tensión y competencia por nuestros recursos limitados?", "relevance": "Identifica los principales impulsores de la tensión del sistema, ayudando a priorizar áreas donde la falta de recursos tendría las consecuencias más graves.", "formula": r'''\text{Tensión}_i = \text{Riesgo}_i \times \text{IncidentesEsperados}_i'''},
                "Sensibilidad al Caos": {"description": "Una medida de la volatilidad y fragilidad de todo el sistema, basada en la Teoría del Caos.", "question": "¿Está la ciudad operando normalmente, o está en un estado 'frágil' donde un pequeño incidente podría desencadenar una crisis mayor?", "relevance": "Actúa como una 'alarma de inestabilidad' crítica. Un puntaje alto advierte que todo el sistema es volátil.", "formula": r'''\lambda \approx \frac{1}{t} \ln \frac{\|\delta(t)\|}{\|\delta(0)\|}'''},
                "Puntaje de Anomalía": {"description": "Mide la 'extrañeza' del patrón actual de incidentes en comparación con la norma histórica.", "question": "¿Estamos viendo tipos de incidentes inusuales o incidentes normales en lugares muy inusuales?", "relevance": "Detecta amenazas novedosas que las métricas simples basadas en volumen pasarían por alto. Un puntaje alto es una señal clara de que 'hoy no es un día normal'.", "formula": r'''D_{KL}(P || Q) = \sum_{z} P(z) \log\frac{P(z)}{Q(z)}'''},
                "Índice de Suficiencia de Recursos": {"description": "Una relación a nivel de sistema entre las unidades disponibles y la demanda total esperada, penalizada por la tensión hospitalaria y el tráfico.", "question": "En general, ¿tiene mi sistema suficientes recursos para manejar la demanda prevista para la próxima hora?", "relevance": "Proporciona una métrica de alto nivel para que el mando entienda la capacidad general del sistema.", "formula": r'''\text{RAI} = \frac{\text{UnidadesDisponibles}}{\sum E_i \times (1 + k_{\text{tensión}})}'''},
            }
            for kpi, content in kpi_defs.items():
                st.markdown(f"**{kpi}**"); st.markdown(f"*{content['description']}*")
                cols = st.columns([1, 2])
                with cols[0]:
                    st.markdown("**Pregunta que Responde:**"); st.markdown(f"> {content['question']}")
                    st.markdown("**Relevancia Estratégica:**"); st.markdown(f"> {content['relevance']}")
                with cols[1]:
                    st.markdown("**Formulación Matemática:**"); st.latex(content['formula'])
                st.markdown("---")

    # --- BARRA LATERAL Y ACCIONES ---
    def _render_sidebar(self):
        st.sidebar.title("Controles Estratégicos")
        st.sidebar.markdown("Ajuste los factores de tiempo real para simular diferentes escenarios.")
        new_env = self._build_env_factors_from_sidebar()
        if new_env != st.session_state.env_factors:
            logger.info("Factores ambientales actualizados.")
            st.session_state.env_factors = new_env
            if not st.session_state.simulation_mode: st.rerun()

        st.sidebar.divider()
        st.sidebar.header("Simulación de Escenarios")
        st.session_state.simulation_mode = st.sidebar.toggle("Activar Modo de Simulación", value=st.session_state.get('simulation_mode', False), help="Active para establecer manualmente los recuentos de incidentes y ambulancias. Desactive para volver a los datos en vivo.")
        with st.sidebar.expander("Controles de Simulación", expanded=st.session_state.simulation_mode):
            is_disabled = not st.session_state.simulation_mode
            st.number_input("Establecer Número de Incidentes Activos", min_value=0, value=len(st.session_state.current_incidents), step=1, key="sim_incident_count", disabled=is_disabled)
            st.number_input("Establecer Número de Ambulancias Disponibles", min_value=0, max_value=len(self.dm.ambulances), value=sum(1 for a in self.dm.ambulances.values() if a['status'] == 'Disponible'), step=1, key="sim_ambulance_count", disabled=is_disabled)
            if st.button("Aplicar Escenario", disabled=is_disabled, use_container_width=True):
                with st.spinner("Generando nuevo escenario..."):
                    new_amb_count = st.session_state.sim_ambulance_count
                    for i, amb_id in enumerate(sorted(self.dm.ambulances.keys())): self.dm.ambulances[amb_id]['status'] = 'Disponible' if i < new_amb_count else 'En Misión'
                    logger.info(f"SIMULACIÓN: Ambulancias disponibles establecidas en {new_amb_count}.")
                    new_inc_count = st.session_state.sim_incident_count
                    st.session_state.current_incidents = self.dm._generate_synthetic_incidents(st.session_state.env_factors, override_count=new_inc_count)
                    logger.info(f"SIMULACIÓN: Generados {new_inc_count} incidentes sintéticos.")
                st.rerun()
        
        st.sidebar.divider()
        st.sidebar.header("Datos y Reportes")
        self._sidebar_file_uploader()
        if st.sidebar.button("Generar y Descargar Reporte PDF", use_container_width=True): self._generate_report()

    def _build_env_factors_from_sidebar(self) -> EnvFactorsWithTolerance:
        env = st.session_state.env_factors
        with st.sidebar.expander("Factores Ambientales Generales", expanded=True):
            is_holiday=st.checkbox("Es Feriado",value=env.is_holiday); weather_options = ["Despejado", "Lluvia", "Niebla"]; weather=st.selectbox("Clima",weather_options,index=weather_options.index(env.weather)); aqi=st.slider("Índice de Calidad del Aire (ICA)",0.0,500.0,env.air_quality_index,5.0); heatwave=st.checkbox("Alerta por Ola de Calor",value=env.heatwave_alert)
        with st.sidebar.expander("Factores Contextuales y de Eventos", expanded=True):
            day_type_options = ['Entre Semana','Viernes','Fin de Semana']
            day_type=st.selectbox("Tipo de Día",day_type_options,index=day_type_options.index(env.day_type));
            time_of_day_options = ['Hora Pico Mañana','Mediodía','Hora Pico Tarde','Noche']
            time_of_day=st.selectbox("Hora del Día",time_of_day_options,index=time_of_day_options.index(env.time_of_day));
            public_event_options = ['Ninguno','Evento Deportivo','Concierto/Festival','Protesta Pública']
            public_event=st.selectbox("Tipo de Evento Público",public_event_options,index=public_event_options.index(env.public_event_type));
            school_in_session=st.checkbox("Clases en Sesión",value=env.school_in_session)
        with st.sidebar.expander("Factores de Tensión del Sistema y Respuesta", expanded=True):
            traffic=st.slider("Nivel General de Tráfico",CONSTANTS['TRAFFIC_MIN'],CONSTANTS['TRAFFIC_MAX'],env.traffic_level,0.1); h_divert=st.slider("Estado de Desvío de Hospitales (%)",0,100,int(env.hospital_divert_status*100),5);
            police_activity_options = ['Bajo','Normal','Alto']
            police_activity=st.selectbox("Nivel de Actividad Policial",police_activity_options,index=police_activity_options.index(env.police_activity))
        return EnvFactorsWithTolerance(is_holiday=is_holiday,weather=weather,traffic_level=traffic,major_event=(public_event!='Ninguno'),population_density=env.population_density,air_quality_index=aqi,heatwave_alert=heatwave,day_type=day_type,time_of_day=time_of_day,public_event_type=public_event,hospital_divert_status=h_divert/100.0,police_activity=police_activity,school_in_session=school_in_session)

    def _sidebar_file_uploader(self):
        up_file = st.sidebar.file_uploader("Cargar Historial de Incidentes (JSON)", type=["json"], key="history_uploader")
        if up_file:
            try:
                data = json.load(up_file)
                if not isinstance(data,list) or not all('location' in d and 'type' in d for d in data): raise ValueError("JSON inválido.")
                st.session_state.historical_data=data; st.sidebar.success(f"Cargados {len(data)} registros históricos."); st.rerun()
            except Exception as e:
                logger.error(f"Error al cargar el archivo: {e}", exc_info=True); st.sidebar.error(f"Error al cargar los datos: {e}")

    def _generate_report(self):
        with st.spinner("Generando Reporte..."):
            try:
                pdf_buffer = ReportGenerator.generate_pdf_report(kpi_df=st.session_state.kpi_df, forecast_df=st.session_state.forecast_df, allocations=st.session_state.allocations, env_factors=st.session_state.env_factors)
                if pdf_buffer.getbuffer().nbytes > 0:
                    st.sidebar.download_button(label="Descargar Reporte PDF",data=pdf_buffer,file_name=f"Reporte_Phoenix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",mime="application/pdf",use_container_width=True)
                else: raise ValueError("El buffer del PDF generado está vacío.")
            except Exception as e:
                logger.error(f"Fallo en la generación del reporte: {e}", exc_info=True); st.sidebar.error(f"Fallo en la generación del reporte: {e}")

def main():
    """Función principal para inicializar y ejecutar la aplicación."""
    try:
        config = load_config(); data_manager = DataManager(config); engine = PredictiveAnalyticsEngine(data_manager, config); dashboard = Dashboard(data_manager, engine); dashboard.render()
    except Exception as e:
        logger.critical(f"Ocurrió un error fatal durante el inicio de la aplicación: {e}", exc_info=True)
        st.error(f"Ocurrió un error fatal en la aplicación: {e}. Por favor, revise los registros y el archivo de configuración.")

if __name__ == "__main__":
    main()
