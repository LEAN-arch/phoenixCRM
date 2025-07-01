# core.py
"""
Core business logic for the RedShield AI Phoenix application.

This module contains the primary classes responsible for data management,
predictive analytics, and resource allocation optimization. It is designed to be
robust, extensible, and performant, with clear separation of concerns.

- DataManager: Handles all data ingestion, cleaning, and preparation from various
  sources (config files, real-time APIs).
- PredictiveAnalyticsEngine: The heart of the system, orchestrating a suite of
  statistical, machine learning, and heuristic models to generate comprehensive
  risk KPIs for each operational zone.
- EnvFactors: A structured data class for environmental and contextual variables.
- TCNN: A deep learning model for advanced time-series forecasting (optional).
"""

import io
import json
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.optimize import LinearConstraint, Bounds, milp, minimize
from shapely.geometry import Point, Polygon

from models import AdvancedAnalyticsLayer

# --- Optional Dependency Handling ---
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch
    import torch.nn as nn

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn: Module = object
    logging.info("PyTorch not found. TCNN model will be disabled.")

try:
    from pgmpy.models import BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    class BayesianNetwork: pass
    class TabularCPD: pass
    class VariableElimination: pass
    logging.info("pgmpy not found. Bayesian network will be disabled.")

# --- System Setup ---
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# --- L1: DATA STRUCTURES ---

@dataclass(frozen=True)
class EnvFactors:
    """
    Immutable dataclass to hold all environmental and contextual factors.
    Using frozen=True makes instances hashable, suitable for caching.
    """
    is_holiday: bool
    weather: str
    traffic_level: float
    major_event: bool
    population_density: float
    air_quality_index: float
    heatwave_alert: bool
    day_type: str
    time_of_day: str
    public_event_type: str
    hospital_divert_status: float
    police_activity: str
    school_in_session: bool


# --- L2: DEEP LEARNING MODEL (CONDITIONAL) ---

class TCNN(nn.Module if TORCH_AVAILABLE else object):
    """
    Temporal Convolutional Neural Network for advanced forecasting.
    This model is conditionally available based on PyTorch installation.
    """
    def __init__(self, input_size: int, output_size: int, channels: List[int], kernel_size: int, dropout: float):
        if not TORCH_AVAILABLE:
            self.model = None
            self.output_size = output_size
            return
        super().__init__()
        layers: List[Any] = []
        in_channels = input_size
        for out_channels in channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding='same'),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        layers.extend([
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, output_size)
        ])
        self.model = nn.Sequential(*layers)
        self.output_size = output_size

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass for the TCNN model."""
        if not TORCH_AVAILABLE or self.model is None:
            return torch.zeros(x.shape[0], self.output_size)
        return self.model(x)


# --- L3: CORE LOGIC CLASSES ---

class DataManager:
    """Manages all data loading, validation, and preparation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_config = config['data']
        self.zones = list(self.data_config['zones'].keys())
        self.zones_gdf = self._build_zones_gdf()
        self.road_graph = self._build_road_graph()
        self.ambulances = self._initialize_ambulances()
        self.laplacian_matrix = self._compute_laplacian_matrix()

    def _build_road_graph(self) -> nx.Graph:
        """Constructs the road network graph from configuration."""
        G = nx.Graph()
        G.add_nodes_from(self.zones)
        edges = self.data_config.get('road_network', {}).get('edges', [])
        valid_edges = [
            (u, v, float(w)) for u, v, w in edges
            if u in G.nodes and v in G.nodes and isinstance(w, (int, float)) and w > 0
        ]
        G.add_weighted_edges_from(valid_edges)
        logger.info(f"Road graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G

    def _build_zones_gdf(self) -> gpd.GeoDataFrame:
        """Constructs the GeoDataFrame for operational zones."""
        zone_data = []
        for name, data in self.data_config['zones'].items():
            try:
                poly = Polygon([(lon, lat) for lat, lon in data['polygon']])
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty:
                    raise ValueError(f"Polygon for zone '{name}' is invalid or empty after buffering.")
                zone_data.append({'name': name, 'geometry': poly, **data})
            except Exception as e:
                logger.error(f"Could not load polygon for zone '{name}': {e}. Skipping.", exc_info=True)

        if not zone_data:
            raise RuntimeError("Fatal: No valid zones could be loaded from configuration.")

        gdf = gpd.GeoDataFrame(zone_data, crs="EPSG:4326").set_index('name')
        logger.info(f"Built GeoDataFrame with {len(gdf)} zones.")
        return gdf

    def _initialize_ambulances(self) -> Dict[str, Any]:
        """Initializes ambulance data from configuration with robust error handling."""
        ambulances = {}
        for amb_id, data in self.data_config['ambulances'].items():
            try:
                ambulances[amb_id] = {
                    'id': amb_id,
                    'status': data.get('status', 'Disponible'),
                    'home_base': data.get('home_base'),
                    'location': Point(float(data['location'][1]), float(data['location'][0]))
                }
            except (ValueError, TypeError, KeyError, IndexError) as e:
                logger.error(f"Could not initialize ambulance '{amb_id}': {e}. Skipping.", exc_info=True)
        logger.info(f"Initialized {len(ambulances)} ambulances.")
        return ambulances

    def _compute_laplacian_matrix(self) -> np.ndarray:
        """Computes the normalized graph Laplacian matrix with a robust fallback."""
        try:
            sorted_zones = sorted(self.road_graph.nodes())
            laplacian = nx.normalized_laplacian_matrix(self.road_graph, nodelist=sorted_zones).toarray()
            logger.info("Graph Laplacian computed successfully.")
            return laplacian
        except Exception as e:
            logger.warning(f"Could not compute Graph Laplacian: {e}. Using identity matrix fallback.")
            return np.eye(len(self.zones))

    def get_current_incidents(self, env_factors: EnvFactors) -> List[Dict[str, Any]]:
        """
        Fetches real-time incidents from an API or local file.
        Falls back to generating synthetic data on failure.
        """
        api_config = self.data_config.get('real_time_api', {})
        endpoint = api_config.get('endpoint', '')
        try:
            if endpoint.startswith(('http://', 'https://')):
                headers = {"Authorization": f"Bearer {api_config.get('api_key')}"} if api_config.get('api_key') else {}
                response = requests.get(endpoint, headers=headers, timeout=10)
                response.raise_for_status()
                incidents = response.json().get('incidents', [])
            else:
                with open(endpoint, 'r', encoding='utf-8') as f:
                    incidents = json.load(f).get('incidents', [])

            valid_incidents = self._validate_incidents(incidents)
            return valid_incidents if valid_incidents else self._generate_synthetic_incidents(env_factors)
        except (requests.exceptions.RequestException, FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to get real-time incidents from '{endpoint}': {e}. Falling back to synthetic data.")
            return self._generate_synthetic_incidents(env_factors)
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching incidents: {e}", exc_info=True)
            return self._generate_synthetic_incidents(env_factors)

    def _validate_incidents(self, incidents: List[Dict]) -> List[Dict]:
        """Validates the structure and data types of raw incident data."""
        valid_incidents = []
        for inc in incidents:
            loc = inc.get('location')
            if all(k in inc for k in ['id', 'type', 'triage']) and isinstance(loc, dict) and 'lat' in loc and 'lon' in loc:
                try:
                    inc['location']['lat'] = float(loc['lat'])
                    inc['location']['lon'] = float(loc['lon'])
                    valid_incidents.append(inc)
                except (ValueError, TypeError):
                    logger.warning(f"Skipping incident {inc.get('id', 'N/A')} due to invalid location data.")
        return valid_incidents

# In core.py, inside the DataManager class:

    def _generate_synthetic_incidents(self, env_factors: EnvFactors, override_count: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generates realistic synthetic incidents based on environmental factors."""
        if override_count is not None:
            # If an override count is provided, use it directly.
            num_incidents = override_count
            logger.info(f"Generating {num_incidents} synthetic incidents based on user override.")
        else:
            # Otherwise, calculate the number of incidents based on environmental factors.
            base_intensity = 5.0
            intensity = base_intensity * \
                (1.5 if env_factors.is_holiday else 1.0) * \
                (1.2 if env_factors.weather in ['Rain', 'Fog'] else 1.0) * \
                (2.0 if env_factors.major_event else 1.0)
            num_incidents = int(np.random.poisson(intensity))
            logger.info(f"Generating {num_incidents} synthetic incidents based on environmental factors.")
        
        if num_incidents == 0:
            return []

        city_boundary = self.zones_gdf.union_all()
        bounds = city_boundary.bounds
        incidents = []
        incident_types = list(self.data_config['distributions']['incident_type'].keys())

        # Optimized: Generate points in a vectorized manner and filter
        num_to_generate = int(num_incidents * 1.5) # Generate more to account for those outside boundary
        lons = np.random.uniform(bounds[0], bounds[2], num_to_generate)
        lats = np.random.uniform(bounds[1], bounds[3], num_to_generate)
        points = gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(lons, lats)])
        valid_points = points[points.within(city_boundary)]

        for i, point in enumerate(valid_points.head(num_incidents)):
            incidents.append({
                'id': f"SYN-{i}",
                'type': np.random.choice(incident_types),
                'triage': 'Red',
                'location': {'lat': point.y, 'lon': point.x},
                'timestamp': datetime.utcnow().isoformat()
            })
        
        return incidents

    def generate_sample_history_file(self) -> io.BytesIO:
        """Generates a sample historical data file for user download."""
        default_env = EnvFactors(is_holiday=False, weather="Clear", traffic_level=1.0, major_event=False, population_density=50000, air_quality_index=50.0, heatwave_alert=False, day_type='Weekday', time_of_day='Midday', public_event_type='None', hospital_divert_status=0.0, police_activity='Normal', school_in_session=True)
        sample_data = [{'incidents': self._generate_synthetic_incidents(default_env), 'timestamp': (datetime.utcnow() - timedelta(hours=i*24)).isoformat()} for i in range(3)]
        buffer = io.BytesIO()
        buffer.write(json.dumps(sample_data, indent=2).encode('utf-8'))
        buffer.seek(0)
        return buffer


class PredictiveAnalyticsEngine:
    """Orchestrates foundational and advanced analytics to produce risk scores."""

    def __init__(self, dm: DataManager, config: Dict[str, Any]):
        self.dm = dm
        self.config = config
        self.model_params = config['model_params']
        self.forecast_df = pd.DataFrame()
        self.bn_model = self._build_bayesian_network()
        self.tcnn_model = TCNN(**config['tcnn_params']) if TORCH_AVAILABLE else None
        weights_config = self.model_params.get('ensemble_weights', {})
        total_weight = sum(weights_config.values())
        self.method_weights = {k: v / total_weight for k, v in weights_config.items()} if total_weight > 0 else {}
        self.gnn_structural_risk = AdvancedAnalyticsLayer._calculate_gnn_risk(self.dm.road_graph)

    @st.cache_resource
    def _build_bayesian_network(_self) -> Optional[BayesianNetwork]:
        """Builds and validates the Bayesian Network from config."""
        if not PGMPY_AVAILABLE: return None
        try:
            bn_config = _self.config['bayesian_network']
            model = BayesianNetwork(bn_config['structure'])
            for node, params in bn_config['cpds'].items():
                model.add_cpds(TabularCPD(variable=node, variable_card=params['card'], values=params['values'], evidence=params.get('evidence'), evidence_card=params.get('evidence_card')))
            model.check_model(); logger.info("Bayesian network initialized."); return model
        except Exception as e:
            logger.warning(f"Failed to initialize Bayesian network: {e}. Disabling.", exc_info=True); return None

    @st.cache_data
    def generate_kpis(_self, historical_data: List[Dict], env_factors: EnvFactors, current_incidents: List[Dict]) -> pd.DataFrame:
        """Master method to generate all Key Performance Indicators (KPIs)."""
        kpi_cols = [
            'Incident Probability', 'Expected Incident Volume', 'Risk Entropy', 'Anomaly Score', 
            'Spatial Spillover Risk', 'Resource Adequacy Index', 'Chaos Sensitivity Score', 
            'Bayesian Confidence Score', 'Information Value Index', 'Response Time Estimate', 
            'Trauma Clustering Score', 'Disease Surge Score', 'Trauma-Disease Correlation', 
            'Violence Clustering Score', 'Accident Clustering Score', 'Medical Surge Score', 
            'Ensemble Risk Score', 'STGP_Risk', 'HMM_State_Risk', 'GNN_Structural_Risk', 
            'Game_Theory_Tension', 'Integrated_Risk_Score'
        ]
        kpi_df = pd.DataFrame(0, index=_self.dm.zones, columns=kpi_cols, dtype=float)

        all_incidents = [inc for h in historical_data for inc in h.get('incidents', []) if isinstance(h, dict)] + current_incidents
        if not all_incidents: return kpi_df.reset_index().rename(columns={'index': 'Zone'})

        incident_df = pd.DataFrame(all_incidents).drop_duplicates(subset=['id'], keep='first')
        incident_gdf = gpd.GeoDataFrame(incident_df, geometry=[Point(loc['lon'], loc['lat']) for loc in incident_df['location']], crs="EPSG:4326")
        
        # --- CORRECTED: This is the robust way to perform the spatial join and get the zone name ---
        zones_for_join = _self.dm.zones_gdf.reset_index()
        incidents_with_zones = gpd.sjoin(incident_gdf, zones_for_join, how="inner", predicate="within").rename(columns={'name': 'Zone'})

        if incidents_with_zones.empty: return kpi_df.reset_index().rename(columns={'index': 'Zone'})

        # --- The rest of the function logic is preserved as it was ---
        incident_counts = incidents_with_zones['Zone'].value_counts().reindex(_self.dm.zones, fill_value=0)
        violence_counts = incidents_with_zones[incidents_with_zones['type'] == 'Trauma-Violence']['Zone'].value_counts().reindex(_self.dm.zones, fill_value=0)
        accident_counts = incidents_with_zones[incidents_with_zones['type'] == 'Trauma-Accident']['Zone'].value_counts().reindex(_self.dm.zones, fill_value=0)
        medical_counts = incidents_with_zones[incidents_with_zones['type'].isin(['Medical-Chronic', 'Medical-Acute'])]['Zone'].value_counts().reindex(_self.dm.zones, fill_value=0)
        
        day_time_multiplier = {'Weekday': 1.0, 'Friday': 1.2, 'Weekend': 1.3}.get(env_factors.day_type, 1.0) * {'Morning Rush': 1.1, 'Midday': 0.9, 'Evening Rush': 1.2, 'Night': 1.4}.get(env_factors.time_of_day, 1.0)
        event_multiplier = {'Sporting Event': 1.6, 'Concert/Festival': 1.8, 'Public Protest': 2.0}.get(env_factors.public_event_type, 1.0) if env_factors.public_event_type != 'None' else 1.0
        violence_event_mod = {'Sporting Event': 1.8, 'Public Protest': 2.5}.get(env_factors.public_event_type, 1.0)
        medical_event_mod = {'Concert/Festival': 2.0}.get(env_factors.public_event_type, 1.0)
        effective_traffic = env_factors.traffic_level * (1.0 if env_factors.school_in_session else 0.8)
        police_activity_mod = {'Low': 1.1, 'Normal': 1.0, 'High': 0.85}.get(env_factors.police_activity, 1.0)
        system_strain_penalty = 1.0 + (env_factors.hospital_divert_status * 2.0)
        
        if _self.bn_model:
            try:
                inference = VariableElimination(_self.bn_model)
                evidence = {'Holiday':1 if env_factors.is_holiday else 0, 'Weather':1 if env_factors.weather!='Clear' else 0, 'MajorEvent':1 if env_factors.major_event else 0, 'AirQuality':1 if env_factors.air_quality_index>100 else 0, 'Heatwave':1 if env_factors.heatwave_alert else 0}
                result = inference.query(variables=['IncidentRate'], evidence=evidence, show_progress=False)
                rate_probs = result.values
                baseline_rate = np.sum(rate_probs * np.array([1, 5, 10])); kpi_df['Bayesian Confidence Score'] = 1 - (np.std(rate_probs) / (np.mean(rate_probs) + 1e-9))
            except Exception as e: logger.warning(f"BNI failed: {e}"); baseline_rate, kpi_df['Bayesian Confidence Score'] = 5.0, 0.5
        else: baseline_rate, kpi_df['Bayesian Confidence Score'] = 5.0, 0.5
        
        baseline_rate *= day_time_multiplier * event_multiplier
        current_dist = incident_counts / (incident_counts.sum() + 1e-9); prior_dist = pd.Series(_self.config['data']['distributions']['zone']).reindex(_self.dm.zones, fill_value=1e-9)
        kpi_df['Anomaly Score'] = np.nansum(current_dist * np.log(current_dist / prior_dist)); kpi_df['Risk Entropy'] = -np.nansum(current_dist * np.log2(current_dist.replace(0,1e-9)))
        kpi_df['Chaos Sensitivity Score'] = _self._calculate_lyapunov_exponent(historical_data)
        base_probs = (baseline_rate * prior_dist * _self.dm.zones_gdf['crime_rate_modifier']).clip(0, 1)
        kpi_df['Spatial Spillover Risk'] = _self.model_params['laplacian_diffusion_factor'] * (_self.dm.laplacian_matrix @ base_probs.values)
        hawkes, sir = _self.model_params['hawkes_process'], _self.model_params['sir_model']
        kpi_df['Violence Clustering Score'] = (violence_counts * hawkes['kappa'] * hawkes['violence_weight'] * violence_event_mod * police_activity_mod).clip(0, 1)
        kpi_df['Accident Clustering Score'] = (accident_counts * hawkes['kappa'] * hawkes['trauma_weight'] * effective_traffic).clip(0, 1)
        kpi_df['Medical Surge Score'] = (_self.dm.zones_gdf['population'].apply(lambda s: sir['beta']*medical_counts.get(s,0)/(s+1e-9)-sir['gamma'])*medical_event_mod).clip(0,1)
        kpi_df['Trauma Clustering Score'] = (kpi_df['Violence Clustering Score'] + kpi_df['Accident Clustering Score']) / 2
        kpi_df['Disease Surge Score'] = kpi_df['Medical Surge Score']
        kpi_df['Incident Probability'] = base_probs; kpi_df['Expected Incident Volume'] = (base_probs * 10 * effective_traffic).round()
        available_units = sum(1 for a in _self.dm.ambulances.values() if a['status']=='Disponible')
        kpi_df['Resource Adequacy Index'] = (available_units / (kpi_df['Expected Incident Volume'].sum() * system_strain_penalty + 1e-9)).clip(0, 1)
        kpi_df['Response Time Estimate'] = (10.0 * system_strain_penalty) * (1 + _self.model_params['response_time_penalty'] * (1-kpi_df['Resource Adequacy Index']))
        
        kpi_df['Ensemble Risk Score'] = _self._calculate_ensemble_risk_score(kpi_df, historical_data)
        kpi_df['Information Value Index'] = kpi_df['Ensemble Risk Score'].std()
        kpi_df['STGP_Risk'] = AdvancedAnalyticsLayer._calculate_stgp_risk(incidents_with_zones, _self.dm.zones_gdf)
        kpi_df['HMM_State_Risk'] = AdvancedAnalyticsLayer._calculate_hmm_risk(kpi_df)
        kpi_df['GNN_Structural_Risk'] = _self.gnn_structural_risk
        kpi_df['Game_Theory_Tension'] = AdvancedAnalyticsLayer._calculate_game_theory_tension(kpi_df)
        
        adv_weights = _self.model_params.get('advanced_model_weights', {})
        kpi_df['Integrated_Risk_Score'] = (
            adv_weights.get('base_ensemble', 0.6) * kpi_df['Ensemble Risk Score'] + adv_weights.get('stgp', 0.1) * kpi_df['STGP_Risk'] +
            adv_weights.get('hmm', 0.1) * kpi_df['HMM_State_Risk'] + adv_weights.get('gnn', 0.1) * kpi_df['GNN_Structural_Risk'] +
            adv_weights.get('game_theory', 0.1) * kpi_df['Game_Theory_Tension']
        ).clip(0, 1)

        return kpi_df.fillna(0).reset_index().rename(columns={'index': 'Zone'})

    def generate_kpis_with_sparklines(self, historical_data: List[Dict], env_factors: EnvFactors, current_incidents: List[Dict]) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
        """Wrapper that generates KPIs and simulated historical trends for UI gauges."""
        kpi_df = self.generate_kpis(historical_data, env_factors, current_incidents)
        sparkline_data = {}
        def create_gauge_data(current_value, history_generator):
            values = np.append(history_generator, current_value).tolist()
            p10, p90 = np.percentile(values, 10), np.percentile(values, 90)
            return {'values': values, 'range': [p10, p90]}
        inc_count = len(current_incidents)
        inc_hist = np.clip(inc_count + np.random.randn(23) * 2 + np.sin(np.linspace(0, np.pi, 23)) * 3, 0, None).astype(int)
        sparkline_data['active_incidents'] = create_gauge_data(inc_count, inc_hist)
        amb_count = sum(1 for a in self.dm.ambulances.values() if a['status'] == 'Disponible')
        amb_hist = np.clip(amb_count + np.random.randn(23), 0, None).astype(int)
        sparkline_data['available_ambulances'] = create_gauge_data(amb_count, amb_hist)
        max_risk = kpi_df['Integrated_Risk_Score'].max()
        risk_hist = np.clip(max_risk + np.random.randn(23) * 0.05, 0, 1)
        sparkline_data['max_risk'] = create_gauge_data(max_risk, risk_hist)
        adequacy = kpi_df['Resource Adequacy Index'].mean()
        adeq_hist = np.clip(adequacy + np.random.randn(23) * 0.03, 0, 1)
        sparkline_data['adequacy'] = create_gauge_data(adequacy, adeq_hist)
        return kpi_df, sparkline_data
        
    def _calculate_lyapunov_exponent(self, historical_data: List[Dict]) -> float:
        """Calculates a proxy for the Lyapunov exponent to measure system chaos."""
        if len(historical_data) < 2: return 0.0
        try:
            series = pd.Series([len(h.get('incidents', [])) for h in historical_data])
            if len(series) < 10 or series.std() == 0: return 0.0
            return np.log(series.diff().abs().mean() + 1)
        except Exception: return 0.0

    def _calculate_ensemble_risk_score(self, kpi_df: pd.DataFrame, historical_data: List[Dict]) -> pd.Series:
        """Blends foundational model outputs into a robust ensemble score."""
        if kpi_df.empty or not self.method_weights: return pd.Series(0.0, index=kpi_df.index)
        norm_df = pd.DataFrame(index=kpi_df.index)
        def normalize(s):
            m, M = s.min(), s.max()
            return (s - m) / (M - m) if M > m else pd.Series(0.0, index=s.index)
        chaos_amp = self.model_params.get('chaos_amplifier', 1.5) if historical_data and np.var([len(h.get('incidents',[])) for h in historical_data]) > np.mean([len(h.get('incidents',[])) for h in historical_data]) else 1.0
        cmap = {'hawkes':'Trauma Clustering Score','sir':'Disease Surge Score','bayesian':'Bayesian Confidence Score','graph':'Spatial Spillover Risk','chaos':'Chaos Sensitivity Score','info':'Risk Entropy','game':'Resource Adequacy Index','violence':'Violence Clustering Score','accident':'Accident Clustering Score','medical':'Medical Surge Score'}
        for wk, m in cmap.items():
            if m in kpi_df.columns and self.method_weights.get(wk, 0) > 0:
                col = kpi_df[m].copy()
                if m == 'Resource Adequacy Index': col = 1 - col
                if m == 'Chaos Sensitivity Score': col *= chaos_amp
                norm_df[wk] = normalize(col)
        if self.method_weights.get('tcnn', 0) > 0 and not self.forecast_df.empty:
            tcnn_risk = self.forecast_df[self.forecast_df['Horizon (Hours)'] == 3].set_index('Zone')[['Violence Risk', 'Accident Risk', 'Medical Risk']].mean(axis=1)
            norm_df['tcnn'] = normalize(tcnn_risk.reindex(self.dm.zones, fill_value=0))
        weights = pd.Series(self.method_weights)
        aligned_scores, aligned_weights = norm_df.align(weights, axis=1, fill_value=0)
        return aligned_scores.dot(aligned_weights).clip(0, 1)

    def generate_forecast(self, kpi_df: pd.DataFrame) -> pd.DataFrame:
        """Generates a risk forecast for all zones."""
        if kpi_df.empty: return pd.DataFrame()
        forecast_data = []
        for _, row in kpi_df.iterrows():
            for horizon in self.config['forecast_horizons_hours']:
                decay = self.model_params['fallback_forecast_decay_rates'].get(str(horizon), 0.5)
                risk = row.get('Integrated_Risk_Score', row['Ensemble Risk Score']) * decay
                unc = risk * np.random.uniform(0.15, 0.25)
                forecast_data.append({
                    'Zone': row['Zone'], 'Horizon (Hours)': horizon, 'Combined Risk': risk,
                    'Upper_Bound': np.clip(risk + unc, 0, 1), 'Lower_Bound': np.clip(risk - unc, 0, 1),
                    'Violence Risk': row.get('Violence Clustering Score', 0) * decay, 
                    'Accident Risk': row.get('Accident Clustering Score', 0) * decay, 
                    'Medical Risk': row.get('Medical Surge Score', 0) * decay})
        fc_df = pd.DataFrame(forecast_data)
        if fc_df.empty: self.forecast_df = fc_df; return self.forecast_df
        cols = ['Violence Risk', 'Accident Risk', 'Medical Risk', 'Combined Risk', 'Upper_Bound', 'Lower_Bound']
        fc_df[cols] = fc_df[cols].clip(0, 1)
        self.forecast_df = fc_df
        return self.forecast_df
        
    def _post_process_allocations(self, allocations: Dict[str, Any], available_units: int, sort_key: pd.Series) -> Dict[str, int]:
        """Rounds allocations and adjusts to match total available units."""
        final = {z: int(round(v)) for z, v in allocations.items()}
        diff = available_units - sum(final.values())
        if diff != 0:
            order = sort_key.sort_values(ascending=False).index.tolist()
            if diff > 0:
                for i in range(diff): final[order[i % len(order)]] += 1
            else:
                for i in range(abs(diff)):
                    if final[z := order[-(i + 1)]] > 0: final[z] -= 1
        return final

    def _allocate_proportional(self, kpi_df: pd.DataFrame, available_units: int) -> Dict[str, int]:
        """Allocates units proportionally to the integrated risk score."""
        logger.info("Using Proportional Allocation strategy.")
        risk = kpi_df.set_index('Zone')['Integrated_Risk_Score']
        if risk.sum() == 0:
            alloc = {z: available_units // len(self.dm.zones) for z in self.dm.zones}
            alloc[self.dm.zones[0]] += available_units % len(self.dm.zones)
            return alloc
        alloc = (available_units * risk / risk.sum()).to_dict()
        return self._post_process_allocations(alloc, available_units, risk)

    def _allocate_milp(self, kpi_df: pd.DataFrame, available_units: int) -> Dict[str, int]:
        """Allocates units using Mixed-Integer Linear Programming."""
        logger.info("Using MILP (Linear Optimization) strategy.")
        zones, risk = kpi_df['Zone'].tolist(), kpi_df['Integrated_Risk_Score'].values
        res = milp(c=-risk, constraints=LinearConstraint(np.ones((1, len(zones))), lb=available_units, ub=available_units), integrality=np.ones_like(risk), bounds=(0, available_units))
        if res.success: return dict(zip(zones, res.x.astype(int)))
        logger.warning("MILP optimization failed. Falling back to proportional.")
        return self._allocate_proportional(kpi_df, available_units)

    def _allocate_nlp(self, kpi_df: pd.DataFrame, available_units: int) -> Dict[str, int]:
        """Allocates using Non-Linear Programming."""
        logger.info("Using NLP (Non-Linear Optimization) strategy.")
        zones, risk, incidents = kpi_df['Zone'].tolist(), kpi_df['Integrated_Risk_Score'].values, kpi_df['Expected Incident Volume'].values
        def obj_func(alloc):
            w_risk, w_cong = self.model_params.get('nlp_weight_risk', 1.0), self.model_params.get('nlp_weight_congestion', 0.2)
            utility = np.sum(risk * np.log(1 + alloc + 1e-9))
            penalty = np.sum(np.square(incidents / (1 + alloc)))
            return -w_risk * utility + w_cong * penalty
        init_guess = (available_units * risk / (risk.sum() + 1e-9)).clip(0)
        res = minimize(fun=obj_func, x0=init_guess, method='SLSQP', bounds=Bounds(lb=0, ub=available_units), constraints=LinearConstraint(np.ones(len(zones)), lb=available_units, ub=available_units))
        if res.success:
            alloc_float = pd.Series(res.x, index=zones)
            return self._post_process_allocations(alloc_float.to_dict(), available_units, alloc_float)
        logger.warning("NLP optimization failed. Falling back to proportional.")
        return self._allocate_proportional(kpi_df, available_units)

    def generate_allocation_recommendations(self, kpi_df: pd.DataFrame) -> Dict[str, int]:
        """Generates optimal resource allocation recommendations."""
        if kpi_df.empty or kpi_df['Integrated_Risk_Score'].sum() == 0:
            return {zone: 0 for zone in self.dm.zones}
        units = sum(1 for a in self.dm.ambulances.values() if a['status'] == 'Disponible')
        if units == 0: return {zone: 0 for zone in self.dm.zones}
        strategy = self.model_params.get('allocation_strategy', 'proportional')
        if strategy == 'nlp': return self._allocate_nlp(kpi_df, units)
        if strategy == 'milp': return self._allocate_milp(kpi_df, units)
        return self._allocate_proportional(kpi_df, units)
