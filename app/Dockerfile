FROM python:3.7
COPY . /application
WORKDIR /application/app
RUN pip install -r app/requirements.txt
EXPOSE $PORT
CMD gunicorn --workers=4 --preload --bind 0.0.0.0:$PORT wsgi:application/app