# read version from installed package
from importlib.metadata import version
import logging
logging.basicConfig(level=logging.DEBUG, 
                    format="%(asctime)s - %(levelname)s - %(message)s"
                    )
__package__ = __file__
logger = logging.getLogger(__name__)
__version__ = version(__package__)
logger.info(f"{__package__} version {__version__}")



