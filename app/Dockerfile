FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip install -r app/requirements.txt
# RUN python -m nltk.downloader punkt

EXPOSE $PORT

CMD gunicorn app:app --workers=4 --preload --bind 0.0.0.0:$PORT 