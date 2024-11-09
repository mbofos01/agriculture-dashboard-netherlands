#!/bin/bash
# This script waits for PostgreSQL to be available before starting the server
# it checks if the necessary tables (CBS, QCL, Weather) exist in the database
# if they do not exist, it waits until they are created
# it then starts the server
# Author: Michail Panagiotis Bofos

# Function to log with timestamp
log_with_timestamp() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo " [$timestamp] - [S] $message"
}

# Function to check if PostgreSQL is reachable
check_postgres() {
    PGPASSWORD=infomdss psql -h "database" -U "student" -d "dashboard" -c '\q' &> /dev/null
}

# Wait until PostgreSQL is available
log_with_timestamp "Checking PostgreSQL connection..."
until check_postgres; do
    log_with_timestamp "PostgreSQL is unavailable - sleeping"
    sleep 5
done

log_with_timestamp "PostgreSQL is up - checking for tables..."

# Function to wait for a specific table to exist
wait_for_table() {
    local table_name="$1"
    until PGPASSWORD=infomdss psql -h "database" -U "student" -d "dashboard" -c "SELECT 1 FROM \"$table_name\" LIMIT 1;" &> /dev/null; do
        log_with_timestamp "Waiting for $table_name table to be created in PostgreSQL..."
        sleep 60
    done
    log_with_timestamp "$table_name table has been created"
}

# Check for each table
wait_for_table "Weather"
wait_for_table "MonthlyWeather"
wait_for_table "CBS"
wait_for_table "QCL"

log_with_timestamp "All tables have been created - starting server..."
/usr/local/bin/python app.py