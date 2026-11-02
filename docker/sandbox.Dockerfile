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
FROM python:3.12-slim

RUN pip install --no-cache-dir "pydantic>=2.6,<3" "typing-extensions>=4.14" \
    && rm -rf /root/.cache

USER 65534:65534
WORKDIR /work
