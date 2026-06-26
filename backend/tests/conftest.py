"""Put backend/ on sys.path so tests can import the engine modules
(``store``, ``layers``, ``codegen``, ``validator``) with bare imports.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
