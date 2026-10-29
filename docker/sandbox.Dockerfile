# Image for orkexa.dynamic.DockerSandbox.
#
#   docker build -t orkexa-sandbox -f docker/sandbox.Dockerfile .
#
# Only the dependencies needed to derive tool schemas and run generated code.
# The orkexa package itself is mounted read-only at run time (/opt/orkexa/orkexa),
# and the container runs without network, with a read-only file system and as
# nobody. No API key or provider SDK is ever installed or passed into it.
FROM python:3.12-slim

RUN pip install --no-cache-dir "pydantic>=2.6,<3" "typing-extensions>=4.14" \
    && rm -rf /root/.cache

USER 65534:65534
WORKDIR /work
