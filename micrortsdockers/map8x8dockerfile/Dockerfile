FROM ubuntu
LABEL LABEL maintainer="Rubens O. Moraes"
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y software-properties-common
RUN apt install -y openjdk-17-jdk openjdk-17-jre
COPY lib /home/MicroRTS/lib
COPY maps /home/MicroRTS/maps
COPY MicroRTS.jar /home/MicroRTS/MicroRTS.jar
COPY logs2 /home/MicroRTS/logs2
RUN chmod +777 /home/MicroRTS/MicroRTS.jar
CMD java -jar /home/MicroRTS/MicroRTS.jar