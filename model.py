# models.py
import pandas as pd
import numpy as np
import geopandas as gpd
import networkx as nx
from typing import Dict, List, Optional, Any

class AdvancedAnalyticsLayer:
    """
    A layer dedicated to calculating KPIs from advanced models.
    These are implemented as computationally inexpensive proxies that capture
    the analytical spirit of each sophisticated technique, demonstrating how they
    would contribute to a holistic risk picture without the full computational cost.
    """
    @staticmethod
    def _calculate_stgp_risk(incidents_with_zones: gpd.GeoDataFrame, zones_gdf: gpd.GeoDataFrame) -> pd.Series:
        """
        Proxy for a Spatiotemporal Gaussian Process (ST-GP).
        This simplified version calculates risk based on geographic proximity to recent 
        high-severity incidents. A real ST-GP would provide a full covariance-based 
        prediction with uncertainty bounds over both space and time.
        """
        stgp_risk = pd.Series(0.0, index=zones_gdf.index)
        if incidents_with_zones.empty:
            return stgp_risk
        
        # Focus on high-severity incidents as the primary drivers of spatial risk
        hotspots = incidents_with_zones[incidents_with_zones['triage'] == 'Red']
        if hotspots.empty:
            return stgp_risk

        zone_centroids = zones_gdf.geometry.centroid
        for zone_name, centroid in zone_centroids.items():
            # Calculate distance-weighted risk from each hotspot to the zone's center
            distances = hotspots.geometry.distance(centroid)
            # Use an exponential decay kernel as a proxy for the GP kernel
            # A smaller distance results in a higher risk score.
            # Length scale is a hyperparameter determining the radius of influence.
            length_scale = 0.05  # Approx. 5km, a reasonable influence radius in degrees
            risk_contribution = np.exp(-0.5 * (distances / length_scale)**2)
            stgp_risk[zone_name] = risk_contribution.sum()
        
        # Normalize the final risk score to a 0-1 scale
        max_risk = stgp_risk.max()
        return (stgp_risk / (max_risk + 1e-9)).clip(0, 1)

    @staticmethod
    def _calculate_hmm_risk(kpi_df: pd.DataFrame) -> pd.Series:
        """
        Proxy for a Hidden Markov Model (HMM).
        A real HMM would use a sequence of observations (e.g., daily incident counts) to infer
        a hidden state (e.g., 'Calm', 'Agitated', 'Critical'). This proxy uses thresholding
        on existing KPIs to simulate these unobserved states.
        """
        # Define the conditions for transitioning to higher-risk states
        is_volatile = kpi_df['Chaos Sensitivity Score'] > 0.5
        is_strained = kpi_df['Resource Adequacy Index'] < 0.5
        is_clustering = kpi_df['Trauma Clustering Score'] > 0.6
        
        # Assign states based on conditions. Higher numbers mean higher risk states.
        hmm_state = pd.Series(0, index=kpi_df.index)  # State 0: Calm
        hmm_state[is_strained] = 1                    # State 1: Strained (resources low)
        hmm_state[is_volatile] = 2                    # State 2: Volatile (unpredictable)
        hmm_state[is_clustering] = 2                  # Also considered volatile
        hmm_state[is_volatile & is_strained] = 3      # State 3: Critical (volatile and strained)
        
        # Normalize the state number to a 0-1 risk score
        return (hmm_state / 3.0).clip(0, 1)

    @staticmethod
    def _calculate_gnn_risk(road_graph: nx.Graph) -> pd.Series:
        """
        Proxy for a Graph Neural Network (GNN).
        A real GNN would learn embeddings for each zone based on its features and connections.
        This proxy uses a pre-computed graph centrality metric as a stand-in for a zone's
        learned structural vulnerability based on its position in the road network.
        """
        # Using betweenness centrality: measures how often a zone lies on the shortest path
        # between other zones. High centrality implies a critical hub.
        centrality = pd.Series(nx.betweenness_centrality(road_graph), name="centrality")
        max_centrality = centrality.max()
        
        # Normalize to get a 0-1 structural risk score
        return (centrality / (max_centrality + 1e-9)).clip(0, 1)

    @staticmethod
    def _calculate_game_theory_tension(kpi_df: pd.DataFrame) -> pd.Series:
        """
        Proxy for a Game Theory model.
        This models the "tension" for resources. A zone's tension is high if it has high
        expected incidents while other zones also have high demand, creating competition
        for a finite pool of ambulances.
        """
        expected_incidents = kpi_df['Expected Incident Volume']
        total_expected = expected_incidents.sum()
        
        if total_expected == 0:
            return pd.Series(0.0, index=kpi_df.index)
            
        # A zone's tension is its share of the total expected demand.
        # It's high when a single zone is responsible for a large fraction of system load.
        tension = expected_incidents / total_expected
        return tension.clip(0, 1)
