FROM neuromation/base:v1.6

ENV ENV_DIR /project

RUN mkdir $ENV_DIR

COPY requirements.txt $ENV_DIR/requirements.txt
COPY apt.txt $ENV_DIR/apt.txt

RUN cd $ENV_DIR && pip install --progress-bar=off -U --no-cache-dir -r requirements.txt
RUN cd $ENV_DIR && export DEBIAN_FRONTEND=noninteractive && \
                   apt-get -qq update && \
                   cat apt.txt | xargs -I % apt-get -qq install --no-install-recommends % && \
                   apt-get -qq clean && \
                   apt-get -qq autoremove && \
                   rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH /project
