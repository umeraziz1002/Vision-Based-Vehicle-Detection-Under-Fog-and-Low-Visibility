"""
app/routes/dashboard.py - Dashboard routes
"""

from flask import Blueprint, render_template
from ..utils.file_utils import get_stats, load_history

dashboard_bp = Blueprint('dashboard', __name__)


@dashboard_bp.route('/')
def dashboard():
    """Main dashboard with statistics and history."""
    stats = get_stats()
    return render_template('dashboard.html', stats=stats)
