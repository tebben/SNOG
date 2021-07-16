import logging

from flask import Flask, make_response, render_template_string
from flask_cors import CORS

from rest import app as snog
from rest.controller.model import app as modelApi
from rest.exceptions import ApiException
from werkzeug.middleware.proxy_fix import ProxyFix

logger = logging.getLogger(__name__)


def create_app():
    """Setup and return the flask app"""

    app = Flask(__name__)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
    app.register_blueprint(snog.apiBlueprint, url_prefix=snog.settings.endpointPrefix)
    app.config["ERROR_404_HELP"] = False

    @app.errorhandler(404)
    def page_not_found(e):
        return make_response(render_template_string('''<!doctype html><html><head><style>*{transition: all 0.6s;}html {height: 100%;}body{font-family: 'Lato', sans-serif;color: #888;margin: 0;}#main{display: table;width: 100%;height: 100vh;text-align: center;}.fof{display: table-cell;vertical-align: middle;}.fof h1{font-size: 50px;display: inline-block;padding-right: 12px;animation: type .5s alternate infinite;}@keyframes type{from{box-shadow: inset -3px 0px 0px #888;}to{box-shadow: inset -3px 0px 0px transparent;}}</style></head><body><div id="main"><div class="fof"><h1>Error 404</h1></div></div></body></html>'''))

    @app.errorhandler(ApiException)
    def handle_api_exception(error):
        logger.info("{0} raised, statusCode: {1}, message: {2}".format(error.__class__.__name__, error.statusCode, error.message))
        return {"message": error.message}, error.statusCode

    if snog.settings.useDefaultCors:
        CORS(app, resources={r"*": {"origins": "*"}})

        @app.after_request
        def after_request(response):
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
            #response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE")
            response.headers.add("Access-Control-Allow-Methods", "GET,POST")
            return response

    return app