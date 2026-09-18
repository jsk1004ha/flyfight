FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8765
WORKDIR /app
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY flyfight ./flyfight
COPY maps ./maps
COPY viewer/FlyFight_Viewer.html ./viewer/FlyFight_Viewer.html
COPY cloud.py ./
USER 10001:10001
EXPOSE 8765
STOPSIGNAL SIGTERM
CMD ["python", "cloud.py"]
