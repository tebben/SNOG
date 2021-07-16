FROM python:3-alpine

ENV SNOG_HOSTNAME=localhost
ENV SNOG_PORT=5000

RUN apk add --update --no-cache g++ gcc libxslt-dev
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt &&\
    addgroup -g 1000 snoggroup &&\
    adduser -u 1000 -G snoggroup -h /home/snoguser -D snoguser
VOLUME /storage
WORKDIR /home/snoguser
USER snoguser
COPY . .
CMD ["sh", "-c", "waitress-serve --listen *:$SNOG_PORT --call rest.manage:create_app"]