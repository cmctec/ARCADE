FROM python:3.8-slim

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app \ 
    && chown user:user /opt/app 


USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools && python -m pip install --user shapely
RUN python -m pip install --user numpy
RUN python -m pip install --user SimpleITK


# COPY --chown=user:user requirements.txt /opt/app/
# RUN python -m piptools sync requirements.txt

# COPY --chown=user:user weights /opt/app/weights
# COPY --chown=user:user weights/model_final.pth /opt/app/

COPY --chown=user:user evaluate.py /opt/app/ 
COPY --chown=user:user saved_images /opt/app/saved_images

ENTRYPOINT [ "python", "-m", "evaluate" ]
