
FROM python:3.12.1-slim

WORKDIR /api

ADD ./api/requirements.txt ./
RUN pip install -r requirements.txt

ADD ./api ./

CMD ["fastapi", "run", "main.py"]
