# Image for orkexa.dynamic.DockerSandbox.
#
#   docker build -t orkexa-sandbox -f docker/sandbox.Dockerfile .
#
# Only the dependencies needed to derive tool schemas and run generated code.
# The tool code, the payload and the few orkexa modules of the sandbox runner travel
# on stdin into a tmpfs (nothing is mounted; the deprecated transport="bind" mounts
# the package at /opt/orkexa/orkexa). The container runs without network, with a
# read-only file system and as nobody. No API key or provider SDK is ever installed or
# passed into it. Check a host with: python -m orkexa.dynamic.docker --self-test
#
# Supply chain: the base image is pinned by the digest of its multi-arch index
# (python:3.12-slim, Python 3.12.16, linux/amd64 and linux/arm64 among others; verified
# against registry-1.docker.io on 2026-11-02), and the Python packages are installed
# from docker/sandbox-requirements.txt with --require-hashes, wheels only. To update the
# base, look up the new index digest (docker buildx imagetools inspect python:3.12-slim)
# and replace it here; tests/test_dockerfiles.py refuses a FROM without a digest.
FROM python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30

#
# Behind a proxy that intercepts TLS, pass its CA certificate (PEM) as the build secret
# "pip-ca"; it is added to pip's CA bundle for this step only and is not part of any layer:
#   docker build --secret id=pip-ca,src=/path/to/proxy-ca.pem \
#     --build-arg HTTPS_PROXY="$HTTPS_PROXY" -t orkexa-sandbox -f docker/sandbox.Dockerfile .
# Without the secret, pip uses its own CA bundle. (BuildKit, the default builder of Docker 23
# and later, and Podman/Buildah support build secrets.)
COPY docker/sandbox-requirements.txt /tmp/sandbox-requirements.txt
RUN --mount=type=secret,id=pip-ca,target=/run/secrets/pip-ca \
    if [ -s /run/secrets/pip-ca ]; then \
        cat "$(python -c 'import pip._vendor.certifi as c; print(c.where())')" /run/secrets/pip-ca > /tmp/pip-ca.pem \
        && export PIP_CERT=/tmp/pip-ca.pem; \
    fi \
    && pip install --no-cache-dir --require-hashes --only-binary=:all: -r /tmp/sandbox-requirements.txt \
    && rm -rf /root/.cache /tmp/sandbox-requirements.txt /tmp/pip-ca.pem

USER 65534:65534
WORKDIR /work
