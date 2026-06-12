"""
run.py - Application entry point
Vision-Based Vehicle Detection under Fog and Low Visibility Conditions
"""

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
