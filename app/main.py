# Welcome to the Flask-Bootstrap sample application. This will give you a
# guided tour around creating an application using Flask-Bootstrap.
#
# To run this application yourself, please install its requirements first:
#
#   $ pip install -r sample_app/requirements.txt
#
# Then, you can actually run the application.
#
#   $ flask --app=sample_app dev
#
# Afterwards, point your browser to http://localhost:5000, then check out the
# source.

from flask import Flask
from flask_appconfig import AppConfig
from flask_bootstrap import Bootstrap

from frontend import frontend_app
from nav import nav_app


app = Flask(__name__)

    # We use Flask-Appconfig here, but this is not a requirement
AppConfig(app)

# Install our Bootstrap extension
Bootstrap(app)

# Our application uses blueprints as well; these go well with the
# application factory. We already imported the blueprint, now we just need
# to register it:
app.register_blueprint(frontend_app)

# Because we're security-conscious developers, we also hard-code disabling
# the CDN support (this might become a default in later versions):
app.config['BOOTSTRAP_SERVE_LOCAL'] = True

# We initialize the navigation as well
nav_app.init_app(app)
