import logging
import time
from datetime import datetime, timezone

def setup_logging(name: str):
    """
    Logging helper:
    - DOES NOT create its own log file
    - ONLY writes start/end markers using the existing root logger
    - Assumes logging.basicConfig() was already called by the script
    """

    start_time = time.time()
    start_timestamp = datetime.now(timezone.utc).isoformat()

    logging.info("==============================================================")
    logging.info(f"Execution started for script: {name}")
    logging.info(f"Start timestamp (UTC): {start_timestamp}")
    logging.info("==============================================================")

    def finish():
        end_timestamp = datetime.now(timezone.utc).isoformat()
        duration = time.time() - start_time
        mins, secs = divmod(duration, 60)

        logging.info("==============================================================")
        logging.info(f"Execution finished for script: {name}")
        logging.info(f"End timestamp (UTC):   {end_timestamp}")
        logging.info(f"Total runtime:         {mins:.0f} min {secs:.2f} s")
        logging.info("==============================================================")

    return finish
