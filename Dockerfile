## Global Args #################################################################
ARG BASE_UBI_IMAGE_TAG=latest
ARG USER=tuning
ARG USER_UID=1000
ARG PYTHON_VERSION=3.11
ARG WHEEL_VERSION=""

## Base Layer ##################################################################
FROM registry.access.redhat.com/ubi9/ubi:${BASE_UBI_IMAGE_TAG} AS base

ARG PYTHON_VERSION
ARG USER
ARG USER_UID

# Note this works for 3.9, 3.11, 3.12
RUN dnf remove -y --disableplugin=subscription-manager \
        subscription-manager \
    && dnf install -y python${PYTHON_VERSION} procps g++ python${PYTHON_VERSION}-devel \
    && ln -s /usr/bin/python${PYTHON_VERSION} /bin/python \
    && python -m ensurepip --upgrade \
    && python -m pip install --upgrade pip \
    && python -m pip install --upgrade setuptools \
    && dnf update -y \
    && dnf clean all

RUN useradd -u $USER_UID ${USER} -m -g 0 --system && \
    chmod g+rx /home/${USER}

FROM base AS python-installations

ARG WHEEL_VERSION
ARG USER
ARG USER_UID

RUN dnf install -y git && \
    # perl-Net-SSLeay.x86_64 and server_key.pem are installed with git as dependencies
    # Twistlock detects it as H severity: Private keys stored in image
    rm -f /usr/share/doc/perl-Net-SSLeay/examples/server_key.pem && \
    dnf clean all

USER ${USER}
WORKDIR /tmp
RUN --mount=type=cache,target=/home/${USER}/.cache/pip,uid=${USER_UID} \
    python -m pip install --user build
COPY --chown=${USER}:root fm_training_estimator fm_training_estimator
COPY .git .git
COPY pyproject.toml pyproject.toml

# Build a wheel if PyPi wheel_version is empty else download the wheel from PyPi
RUN if [[ -z "${WHEEL_VERSION}" ]]; \
    then python -m build --wheel --outdir /tmp; \
    else pip download fm_training_estimator==${WHEEL_VERSION} --dest /tmp --only-binary=:all: --no-deps; \
    fi && \
    ls /tmp/*.whl >/tmp/bdist_name

# Install from the wheel
RUN --mount=type=cache,target=/home/${USER}/.cache/pip,uid=${USER_UID} \
    python -m pip install --user wheel && \
    python -m pip install --user "$(head bdist_name)" && \
    # Cleanup the bdist whl file
    rm $(head bdist_name) /tmp/bdist_name

## Final image ################################################
FROM base AS release
ARG USER
ARG PYTHON_VERSION

RUN mkdir -p /licenses
COPY LICENSE /licenses/

RUN mkdir /app && \
    chown -R $USER:0 /app /tmp && \
    chmod -R g+rwX /app /tmp


# Copy scripts and default configs
COPY launch_estimator.py /app/
RUN chmod +x /app/launch_estimator.py

WORKDIR /app
USER ${USER}
COPY --from=python-installations /home/${USER}/.local /home/${USER}/.local
ENV PYTHONPATH="/home/${USER}/.local/lib/python${PYTHON_VERSION}/site-packages"

CMD [ "python", "/app/launch_estimator.py" ]
