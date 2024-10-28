# Use an official Python runtime as a parent image
FROM python:3.11

ENV PYTHONUNBUFFERED 1

RUN apt-get update

# Install dos2unix
RUN apt-get install -y dos2unix

# Install PostgreSQL client
RUN apt-get update && apt-get install -y postgresql-client 

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Turn all script to unix format
RUN dos2unix /app/*.sh 

# Make wait-for-postgres.sh executable
RUN chmod +x /app/wait-for-postgres.sh

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 8050

# Run the Flask app
ENTRYPOINT ["bash", "/app/wait-for-postgres.sh"]