FROM tensorflow/tensorflow:2.8.0-gpu
RUN useradd -m wjb
RUN chown -R wjb:wjb /home/wjb
COPY --chown=wjb ./requirements.txt /home/wjb/app/
USER wjb
RUN cd /home/wjb/app/ && pip install -r requirements.txt && pip install gym -i https://pypi.doubanio.com/simple
WORKDIR /home/wjb/app