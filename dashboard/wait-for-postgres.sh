#!/bin/bash


# Function to check if PostgreSQL is reachable
check_postgres() {
    PGPASSWORD=infomdss psql -h "database" -U "student" -d "dashboard" -c '\q' &> /dev/null
}

# Wait until PostgreSQL is available
echo "Checking PostgreSQL connection..."
until check_postgres; do
    echo "PostgreSQL is unavailable - sleeping"
    sleep 5
done

echo "PostgreSQL is up - checking for tables..."

# Function to wait for a specific table to exist
wait_for_table() {
    local table_name="$1"
    until PGPASSWORD=infomdss psql -h "database" -U "student" -d "dashboard" -c "SELECT 1 FROM \"$table_name\" LIMIT 1;" &> /dev/null; do
        echo "Waiting for $table_name table to be created in PostgreSQL..."
        sleep 5
    done
    echo "$table_name table has been created"
}

# Check for each table
wait_for_table "Weather"
wait_for_table "CBS"
wait_for_table "QCL"

echo "All tables found! Starting the server."
/usr/local/bin/python app.py