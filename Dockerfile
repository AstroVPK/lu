# Start with Ubuntu base image
FROM ubuntu:16.04

MAINTAINER AstroVPK <vishal.kasliwal@gmail.com>

# Install system tools / libraries
RUN apt-get update && apt-get install -y \
    make \
    g++

ADD Makefile /
ADD main.cc /

# Run the lu decomposition application
ENTRYPOINT ["make", "run-cpu", "COMPILER=gcc"]
CMD ["AUTHOR=vishal"]
