FROM python:3.7.1

ENV SNOG_HOSTNAME=localhost
ENV SNOG_PORT=5000

RUN python --version
RUN pip3 install --upgrade pip

COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt &&\
    groupadd -g 1000 snoggroup &&\
    useradd -rm -d /home/snoguser -s /bin/bash -g root -G snoggroup -u 1000 snoguser
WORKDIR /home/snoguser
USER snoguser
COPY . .
CMD ["sh", "-c", "waitress-serve --listen *:$SNOG_PORT --call rest.manage:create_app"]