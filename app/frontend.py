# This contains our frontend; since it is a bit messy to use the @app.route
# decorator style when using application factories, all of our routes are
# inside blueprints. This is the front-facing blueprint.
#
# You can find out more about blueprints at
# http://flask.pocoo.org/docs/blueprints/

from flask import Blueprint, render_template, flash, redirect, url_for
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
from markupsafe import escape

from nav import nav_app
import json

from machine.Demographic import Demographic


frontend_app = Blueprint('frontend', __name__)

nav_app.register_element('frontend_top', Navbar(
    View('Home', '.index'),
    Subgroup(
        'Demographics', 
        Link('Most Popular', './'),
        Link('Best Score', './most-scored'),
        Link('Most vote count', './most-vote-count'),
        Link('Most vote average', './most-vote-average'),
    ),
    Text('MachineLearning'.format(FLASK_BOOTSTRAP_VERSION)), 
))

# Our index-page just shows a quick explanation. Check out the template
# "templates/index.html" documentation for more details.
@frontend_app.route('/')
def index():
    
    dem = Demographic()
    dem.preprocess(creditsStr='/Dataset/tmdb_5000_credits.csv', moviesStr='/Dataset/tmdb_5000_movies.csv')

    return render_template(
        'index.html',
        data=dem.getMostPopular().to_html(bold_rows=False),
        title="Demographics / Most Popular"
    )


@frontend_app.route('/most-scored')
def most_scored():
    
    dem = Demographic()
    dem.preprocess(creditsStr='/Dataset/tmdb_5000_credits.csv', moviesStr='/Dataset/tmdb_5000_movies.csv')

    return render_template(
        'index.html',
        data=dem.getMostByScored().to_html(bold_rows=False),
        title="Demographics / Most Scored"
    )

@frontend_app.route('/most-vote-count')
def most_vote_count():
    
    dem = Demographic()
    dem.preprocess(creditsStr='/Dataset/tmdb_5000_credits.csv', moviesStr='/Dataset/tmdb_5000_movies.csv')

    return render_template(
        'index.html',
        data=dem.getMostByVoteCount().to_html(bold_rows=False),
        title="Demographics / Most Vote Count"
    )

@frontend_app.route('/most-vote-average')
def most_vote_average():
    
    dem = Demographic()
    dem.preprocess(creditsStr='/Dataset/tmdb_5000_credits.csv', moviesStr='/Dataset/tmdb_5000_movies.csv')

    return render_template(
        'index.html',
        data=dem.getMostByVoteAverage().to_html(bold_rows=False),
        title="Demographics / Most Vote Average"
    )
