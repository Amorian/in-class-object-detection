# Starting with a slim python container
FROM python:3.7.5-slim

# Document who is responsible for this image
MAINTAINER Alankrith Krishnan "alankrith.krishnan@nyu.edu"

# Update and download libglib for opencv
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libsm6 libxext6 libxrender-dev && rm -rf /var/lib/apt/lists/*

# Expose any ports the app is expecting in the environment
ENV PORT 8080
EXPOSE $PORT

# Set up a working folder and install the pre-reqs
WORKDIR /app
ADD requirements.txt /app
RUN pip3 --no-cache-dir install -r requirements.txt

# Add code as the last Docker layer because it changes the most
ADD static /app/static
ADD templates /app/templates
ADD model7.pth /app/model7.pth
ADD test.py  /app/test.py

# Run the service
CMD [ "python", "test.py" ]
