import logging
from shared_storage_service import serve
from tictactoe import make_config

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    serve(config=make_config())
