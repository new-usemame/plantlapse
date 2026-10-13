FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision==0.29.1
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /app
COPY plantlapse /app/plantlapse
ENV HF_HOME=/hf PYTHONUNBUFFERED=1
USER 1000:1000
EXPOSE 5056
CMD ["uvicorn", "plantlapse.api:app", "--host", "0.0.0.0", "--port", "5056", "--no-access-log"]
