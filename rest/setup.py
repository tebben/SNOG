import environ
import logging

from flask import Blueprint
from flask_restx import Api

logger = logging.getLogger(__name__)


@environ.config()
class Settings:
    """App settings loaded from environment variables"""

    version = "0.1"
    title = environ.var("SNOG API", name="SNOG_TITLE")
    description = environ.var("Spatial Multi-objective Optimization of Food-Water-Energy Nexus", name="SNOG_DESCRIPTION")
    endpointPrefix = environ.var("", name="SNOG_ENDPOINT_PREFIX")
    useDefaultCors = environ.var(False, converter=bool, name="SNOG_DEFAULT_CORS")
    logLevel = environ.var("INFO", name="SNOG_LOG_LEVEL")


def setup_logger(logLevel: str):
    """Setup the logger"""

    logLevel = logging.getLevelName(logLevel)

    werkzeug = logging.getLogger('werkzeug')
    waitress = logging.getLogger('waitress')

    werkzeug.setLevel(logLevel)
    waitress.setLevel(logLevel)

    logging.basicConfig(level=logLevel, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z")


class AppConfig:
    """Application config and initialization"""

    def __init__(self):
        # Load config
        self.settings = Settings.from_environ()

        # Setup the logger
        setup_logger(self.settings.logLevel)
        logger.info("Setting up REST API")

        # Setup flask/restx, namespaces
        self.apiBlueprint = Blueprint("api", __name__)
        self.api = Api(self.apiBlueprint, version=self.settings.version, title=self.settings.title, description=self.settings.description)
        self.ns_model = self.api.namespace("model", "SNOG models endpoint")