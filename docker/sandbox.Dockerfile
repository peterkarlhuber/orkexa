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

COPY docker/sandbox-requirements.txt /tmp/sandbox-requirements.txt
RUN pip install --no-cache-dir --require-hashes --only-binary=:all: -r /tmp/sandbox-requirements.txt \
    && rm -rf /root/.cache /tmp/sandbox-requirements.txt

USER 65534:65534
WORKDIR /work
