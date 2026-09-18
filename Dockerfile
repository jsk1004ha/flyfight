# Bootstrap HTTPS trust only; no Python or OS binaries are copied from this stage.
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS certificates
FROM ubuntu:24.04@sha256:b3cc40b72b93588182b5410f723c7aaf142363311c2aa993d8a453ddcbb3ae15 AS runtime
COPY --from=certificates /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
RUN sed -i 's|http://|https://|g' /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update --error-on=any && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends python3.12 libgomp1 ca-certificates \
    && apt-get clean

FROM runtime AS dependencies
RUN apt-get update --error-on=any && apt-get install -y --no-install-recommends python3.12-venv \
    && python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
RUN python -m pip install --no-cache-dir --upgrade pip==26.2.1 setuptools==84.0.0
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# The running service never installs packages. Remove the installer itself,
# including its vendored dependencies; retain runtime package metadata for scanning.
RUN python -m pip check && python -m pip uninstall -y pip

FROM runtime
RUN useradd --uid 10001 --user-group --no-create-home --home-dir /tmp --shell /usr/sbin/nologin flyfight
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8765 PATH="/opt/venv/bin:$PATH" TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor
WORKDIR /app
COPY --from=dependencies /opt/venv /opt/venv
COPY flyfight ./flyfight
COPY maps ./maps
COPY viewer/FlyFight_Viewer.html ./viewer/FlyFight_Viewer.html
COPY cloud.py ./
USER 10001:10001
EXPOSE 8765
STOPSIGNAL SIGTERM
CMD ["python", "cloud.py"]
