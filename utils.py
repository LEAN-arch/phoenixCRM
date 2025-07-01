# utils.py
"""
Utility functions and classes for the RedShield AI Phoenix application.

This module provides helper functionalities including:
- Configuration management: Loading, validating, and saving the system configuration
  from a JSON file, with robust defaults and environment variable overrides.
- PDF Reporting: A class to generate comprehensive PDF situational reports from
  the application's output data (KPIs, forecasts, allocations).
"""

import io
import json
import logging
import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# Assuming core.py is in the same directory for type hinting
from core import EnvFactors

# --- System Setup ---
logger = logging.getLogger(__name__)


def deep_update(source: Dict, overrides: Dict) -> Dict:
    """
    Recursively update a dictionary.
    Unlike dict.update(), this merges nested dictionaries instead of overwriting them.
    """
    for key, value in overrides.items():
        if isinstance(value, Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = value
    return source


def get_default_config() -> Dict[str, Any]:
    """
    Returns the hardcoded default configuration dictionary.
    This serves as the single source of truth for the application's settings.
    """
    return {
        "mapbox_api_key": None,
        "forecast_horizons_hours": [0.5, 1, 3, 6, 12, 24, 72],
        "kpi_columns": [ # Centralized list of all possible KPIs
            'Incident Probability', 'Expected Incident Volume', 'Risk Entropy',
            'Anomaly Score', 'Spatial Spillover Risk', 'Resource Adequacy Index',
            'Chaos Sensitivity Score', 'Bayesian Confidence Score',
            'Information Value Index', 'Response Time Estimate',
            'Trauma Clustering Score', 'Disease Surge Score',
            'Trauma-Disease Correlation', 'Violence Clustering Score',
            'Accident Clustering Score', 'Medical Surge Score',
            'Ensemble Risk Score', 'STGP_Risk', 'HMM_State_Risk',
            'GNN_Structural_Risk', 'Game_Theory_Tension', 'Integrated_Risk_Score'
        ],
        "data": {
            "zones": {
                "Centro": {"polygon": [[32.52, -117.03], [32.54, -117.03], [32.54, -117.05], [32.52, -117.05]], "population": 50000, "crime_rate_modifier": 1.2},
                "Otay": {"polygon": [[32.53, -116.95], [32.54, -116.95], [32.54, -116.98], [32.53, -116.98]], "population": 30000, "crime_rate_modifier": 0.8},
                "Playas": {"polygon": [[32.51, -117.11], [32.53, -117.11], [32.53, -117.13], [32.51, -117.13]], "population": 20000, "crime_rate_modifier": 1.0}
            },
            "ambulances": {
                "A01": {"status": "Disponible", "home_base": "Centro", "location": [32.53, -117.04]},
                "A02": {"status": "Disponible", "home_base": "Otay", "location": [32.535, -116.965]},
                "A03": {"status": "En Misión", "home_base": "Playas", "location": [32.52, -117.12]}
            },
            "distributions": {
                "zone": {"Centro": 0.5, "Otay": 0.3, "Playas": 0.2},
                "incident_type": {"Trauma-Violence": 0.2, "Trauma-Accident": 0.2, "Medical-Chronic": 0.4, "Medical-Acute": 0.2}
            },
            "road_network": {"edges": [["Centro", "Otay", 5], ["Centro", "Playas", 10], ["Otay", "Playas", 8]]},
            "real_time_api": {"endpoint": "sample_api_response.json", "api_key": None}
        },
        "model_params": {
            "hawkes_process": {"kappa": 0.5, "beta": 1.0, "trauma_weight": 1.5, "violence_weight": 1.8},
            "sir_model": {"beta": 0.3, "gamma": 0.1},
            "laplacian_diffusion_factor": 0.1,
            "response_time_penalty": 3.0,
            "hospital_strain_multiplier": 2.0,
            "ensemble_weights": {"hawkes": 0.15, "sir": 0.1, "bayesian": 0.15, "graph": 0.1, "chaos": 0.1, "info": 0.15, "game": 0.25},
            "advanced_model_weights": {"base_ensemble": 0.5, "stgp": 0.15, "hmm": 0.1, "gnn": 0.15, "game_theory": 0.1},
            "chaos_amplifier": 1.5,
            "fallback_forecast_decay_rates": {"0.5": 0.95, "1": 0.9, "3": 0.8, "6": 0.7, "12": 0.6, "24": 0.4, "72": 0.2},
            "allocation_strategy": "nlp", # Options: "proportional", "milp", "nlp"
            "nlp_weight_risk": 1.0,
            "nlp_weight_congestion": 0.2,
        },
        "bayesian_network": {
            "structure": [["Holiday", "IncidentRate"], ["Weather", "IncidentRate"], ["MajorEvent", "IncidentRate"], ["AirQuality", "IncidentRate"], ["Heatwave", "IncidentRate"]],
            "cpds": {
                "Holiday": {"card": 2, "values": [[0.97], [0.03]]},
                "Weather": {"card": 2, "values": [[0.8], [0.2]]},
                "MajorEvent": {"card": 2, "values": [[0.95], [0.05]]},
                "AirQuality": {"card": 2, "values": [[0.8], [0.2]]},
                "Heatwave": {"card": 2, "values": [[0.9], [0.1]]},
                "IncidentRate": {
                    "card": 3,
                    "values": [[0.6,0.5,0.4,0.3,0.5,0.4,0.3,0.2]*4,[0.3,0.3,0.4,0.4,0.3,0.4,0.4,0.5]*4,[0.1,0.2,0.2,0.3,0.2,0.2,0.3,0.3]*4],
                    "evidence": ["Holiday", "Weather", "MajorEvent", "AirQuality", "Heatwave"],
                    "evidence_card": [2, 2, 2, 2, 2]
                }
            }
        },
        "tcnn_params": {"input_size": 8, "output_size": 24, "channels": [16, 32, 64], "kernel_size": 2, "dropout": 0.2}
    }


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validates the configuration dictionary against the default structure.
    Adds missing keys from the default config to ensure robustness.
    Returns True if the configuration was modified, False otherwise.
    """
    modified = False
    default_config = get_default_config()

    # Use deep_update to fill in any missing keys at any level
    original_config_str = json.dumps(config, sort_keys=True)
    config = deep_update(default_config, config)
    updated_config_str = json.dumps(config, sort_keys=True)

    if original_config_str != updated_config_str:
        modified = True
        logger.warning("Configuration was missing keys and has been updated with defaults.")

    return modified


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Loads, validates, and returns the system configuration.
    - Starts with a hardcoded default configuration.
    - Overwrites with settings from `config.json` if it exists.
    - Overwrites with environment variables (e.g., MAPBOX_API_KEY).
    - Validates the final configuration, adding any missing default values.
    - Saves the updated config back to the file if it was modified.
    """
    config = get_default_config()
    config_file = Path(config_path)

    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                # Optimized: Use deep_update to merge user config without losing nested defaults
                config = deep_update(config, user_config)
            logger.info(f"Loaded user configuration from {config_path}.")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {config_path}. Using default configuration.")
        except Exception as e:
            logger.error(f"Failed to load {config_path}: {e}. Using default configuration.", exc_info=True)

    # Override with environment variables
    mapbox_key = os.environ.get("MAPBOX_API_KEY")
    if mapbox_key and "YOUR_KEY" not in mapbox_key:
        config['mapbox_api_key'] = mapbox_key

    # Validate and potentially save back
    if validate_config(config):
        logger.info("Configuration was modified during validation. Saving updated config file.")
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            logger.error(f"Could not save updated configuration to {config_path}: {e}", exc_info=True)

    logger.info("System configuration loaded and validated successfully.")
    return config


class ReportGenerator:
    """
    Handles the generation of PDF reports.
    Refactored for clarity and maintainability.
    """

    @staticmethod
    def _get_styles() -> Dict[str, Any]:
        """Centralizes style definitions for the report."""
        styles = getSampleStyleSheet()
        return {
            'title': styles['Title'],
            'h2': styles['Heading2'],
            'normal': styles['Normal'],
            'env_table': TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (1, -1), colors.lightblue),
            ]),
            'kpi_table': TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkslategray),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('FONTSIZE', (0, 0), (-1, -1), 7),
            ]),
            'alloc_table': TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ])
        }

    @classmethod
    def generate_pdf_report(
        cls, kpi_df: pd.DataFrame, forecast_df: pd.DataFrame,
        allocations: Dict[str, int], env_factors: EnvFactors
    ) -> io.BytesIO:
        """
        Main method to generate the complete PDF report.
        Orchestrates the creation of different report sections.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, title="RedShield AI: Phoenix Situational Report")
        styles = cls._get_styles()
        elements = []

        try:
            cls._add_header(elements, styles)
            cls._add_env_factors_table(elements, styles, env_factors)
            cls._add_kpi_table(elements, styles, kpi_df)
            cls._add_forecast_table(elements, styles, forecast_df)
            cls._add_allocation_table(elements, styles, allocations)

            doc.build(elements)
            buffer.seek(0)
            logger.info("PDF report generated successfully.")
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}", exc_info=True)
            # Return an empty buffer on failure
            return io.BytesIO()

        return buffer

    @staticmethod
    def _add_header(elements: List, styles: Dict):
        """Adds the main title and timestamp to the report."""
        elements.append(Paragraph("RedShield AI: Phoenix v4.0 - Situational Report", styles['title']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['normal']))
        elements.append(Spacer(1, 12))

    @staticmethod
    def _add_env_factors_table(elements: List, styles: Dict, env_factors: EnvFactors):
        """Adds the environmental factors context table."""
        elements.append(Paragraph("Scenario Context: Environmental Factors", styles['h2']))
        env_data = [
            ["Factor", "Value"],
            ["Is Holiday", str(env_factors.is_holiday)],
            ["Weather", str(env_factors.weather)],
            ["Traffic Level", f"{env_factors.traffic_level:.2f}"],
            ["Public Event", str(env_factors.public_event_type)],
            ["AQI", f"{env_factors.air_quality_index:.1f}"],
            ["Heatwave", str(env_factors.heatwave_alert)],
            ["Hospital Strain", f"{env_factors.hospital_divert_status:.0%}"],
            ["Police Activity", str(env_factors.police_activity)]
        ]
        table = Table(env_data, colWidths=[200, 200])
        table.setStyle(styles['env_table'])
        elements.append(table)
        elements.append(Spacer(1, 12))

    @staticmethod
    def _add_kpi_table(elements: List, styles: Dict, kpi_df: pd.DataFrame):
        """Adds the main KPI summary table."""
        elements.append(Paragraph("Risk Analysis Summary", styles['h2']))
        if kpi_df.empty:
            elements.append(Paragraph("No KPI data available.", styles['normal']))
            return

        cols_to_report = [
            'Zone', 'Integrated_Risk_Score', 'Ensemble Risk Score', 'Expected Incident Volume',
            'STGP_Risk', 'HMM_State_Risk', 'GNN_Structural_Risk', 'Game_Theory_Tension'
        ]
        # Robustly select only columns that exist in the DataFrame
        report_cols = [col for col in cols_to_report if col in kpi_df.columns]
        kpi_report_df = kpi_df[report_cols].round(3)

        header = [col.replace('_', ' ').title() for col in kpi_report_df.columns]
        body = kpi_report_df.values.tolist()
        data = [header] + body

        table = Table(data, hAlign='LEFT', repeatRows=1)
        table.setStyle(styles['kpi_table'])
        elements.append(table)
        elements.append(Spacer(1, 12))

    @staticmethod
    def _add_forecast_table(elements: List, styles: Dict, forecast_df: pd.DataFrame):
        """Adds the risk forecast summary table."""
        elements.append(Paragraph("Forecast Summary (Integrated Risk)", styles['h2']))
        if forecast_df.empty:
            elements.append(Paragraph("No forecast data available.", styles['normal']))
            return

        pivot_df = forecast_df.pivot_table(
            index='Zone', columns='Horizon (Hours)', values='Combined Risk'
        ).round(3)

        header = [['Zone'] + [f"{col} hrs" for col in pivot_df.columns]]
        body = [[idx] + row.tolist() for idx, row in pivot_df.iterrows()]
        data = header + body

        table = Table(data, hAlign='LEFT', repeatRows=1)
        table.setStyle(styles['kpi_table']) # Reuse KPI style
        elements.append(table)
        elements.append(Spacer(1, 12))

    @staticmethod
    def _add_allocation_table(elements: List, styles: Dict, allocations: Dict[str, int]):
        """Adds the resource allocation recommendation table."""
        elements.append(Paragraph("Strategic Allocation Recommendation", styles['h2']))
        if not allocations:
            elements.append(Paragraph("No allocation recommendations available.", styles['normal']))
            return

        data = [['Zone', 'Recommended Units']] + list(allocations.items())
        table = Table(data, colWidths=[200, 200])
        table.setStyle(styles['alloc_table'])
        elements.append(table)
