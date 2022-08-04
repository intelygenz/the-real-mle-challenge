FROM python:3.9

ENV PYTHONPATH=lab

WORKDIR /code

COPY ./Pipfile /code/Pipfile
COPY ./Pipfile.lock /code/Pipfile.lock

RUN python -m pip install --upgrade pip
RUN pip install pipenv
RUN pipenv install --clear --system

COPY ./lab/api /code/lab/api
COPY ./models /code/models

CMD ["./lab/api/launch.sh"]