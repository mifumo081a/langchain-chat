FROM python:3.9

RUN apt update
RUN apt-get install  -y zip unzip libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config
ENV TZ=Asia/Tokyo
ENV TERM xterm

# User settings
ARG UID
ARG GID
ARG UNAME
ARG GNAME

RUN echo "Starting with UID: ${UID}, UNAME: ${UNAME}, GID: ${GID}, GNAME: ${GNAME}"
RUN groupadd -g ${GID} -f ${GNAME}
RUN useradd -u ${UID} -g ${GNAME} -m ${UNAME}

COPY ./scripts/setVolumePermission.sh /home/${UNAME}
RUN chmod +x /home/${UNAME}/setVolumePermission.sh

CMD ["sh", "-c", "/home/${UNAME}/setVolumePermission.sh"]

## Install python package
COPY ./requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html
