"""
app/__init__.py - Flask Application Factory
Vision-Based Vehicle Detection under Fog and Low Visibility Conditions
"""

import os
from flask import Flask
from .config import Config


def create_app(config_class=Config):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure required directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    os.makedirs(app.config['LOG_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DEHAZED_FOLDER'], exist_ok=True)

    # NOTE: DehazeFormer model (app/models/dehazer.py + dehazeformer.py) is
    # preserved in code but NOT initialised here — dehazing is disabled in UI.
    # To re-enable: uncomment the two lines below and add the toggle back.
    # from .models.dehazer import dehazer
    # dehazer.set_enabled(app.config['DEHAZER_ENABLED'])

    # Register Blueprints
    from .routes.main import main_bp
    from .routes.detection import detection_bp
    from .routes.dashboard import dashboard_bp
    from .api.endpoints import api_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(detection_bp, url_prefix='/detect')
    app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
    app.register_blueprint(api_bp, url_prefix='/api/v1')

    return app
