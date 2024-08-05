from dotenv import load_dotenv
import os

__all__ = []
_BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

if os.path.exists(os.path.join(_BASE_PATH, ".env")):
    load_dotenv(dotenv_path=os.path.join(_BASE_PATH, ".env"))