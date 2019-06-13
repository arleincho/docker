FROM tiangolo/uwsgi-nginx-flask:python3.7


RUN apt update

RUN apt install -y nano

COPY ./app /app

RUN pip install -r /app/requirements.txt

RUN python /app/manage.py collect

ENV UWSGI_INI /app/uwsgi.ini

RUN cp -R  /app/uwsgi.ini /etc/uwsgi/uwsgi.ini

ENV STATIC_URL /static