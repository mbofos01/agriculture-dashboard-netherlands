# !/bin/bash
# This script is used to periodically check
# for new data from the data sources and
# initiate the ETL process

/usr/local/bin/python /app/sync_faostat.py
/usr/local/bin/python /app/sync_cbs.py