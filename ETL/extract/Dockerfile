FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 

WORKDIR /app
COPY . /app

RUN pip install -r /app/requirements.txt

# Install cron and other dependencies
RUN apt-get update && apt-get install -y cron

# Install dos2unix
RUN apt-get install -y dos2unix

# Turn all script to unix format
RUN dos2unix /app/*.sh 

# Make my_script.sh executable
RUN chmod +x /app/task.sh

# Add cron job
RUN echo "SHELL=/bin/bash\n*/5 * * * * /app/task.sh >> /var/log/cron.log 2>&1" > /etc/cron.d/my_cron

# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/my_cron

# Apply cron job
RUN crontab /etc/cron.d/my_cron

# Create the log file to be able to run tail
RUN touch /var/log/cron.log

# Start cron and stay running
CMD bash task.sh; cron && tail -f /var/log/cron.log
