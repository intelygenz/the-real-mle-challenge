FROM python:3.12-slim

COPY solution/app /the-real-mle-challenge/solution/app
COPY solution/code /the-real-mle-challenge/solution/code
COPY solution/requirements.txt requirements.txt

RUN pip install -r requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:/the-real-mle-challenge/solution/app:/the-real-mle-challenge/solution\
:/the-real-mle-challenge/solution/code:/the-real-mle-challenge/solution/code/src:/the-real-mle-challenge/solution"

WORKDIR /the-real-mle-challenge/solution/app

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]