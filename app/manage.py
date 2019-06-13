# manage.py
#!/usr/bin/env python
import os
from flask_script import Manager, Server
from flask import current_app
from flask_collect import Collect
from main import app

class Config(object):
   # CRITICAL CONFIG VALUE: This tells Flask-Collect where to put our static files!
   # Standard practice is to use a folder named "static" that resides in the top-level of the project directory.
   # You are not bound to this location, however; you may use basically any directory that you wish.
   COLLECT_STATIC_ROOT = os.path.dirname(__file__) + '/static'
   COLLECT_STORAGE = 'flask_collect.storage.file'

# app = create_app(Config)
config = Config
app.config.from_object(config)

manager = Manager(app)
manager.add_command('runserver', Server(host='127.0.0.1', port=5000))

collect = Collect()
collect.init_app(app)

@manager.command
def collect():
  """Collect static from blueprints. Workaround for issue: https://github.com/klen/Flask-Collect/issues/22"""
  return current_app.extensions['collect'].collect()

if __name__ == "__main__":
   manager.run()