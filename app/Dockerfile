FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip install -r app/requirements.txt
EXPOSE $PORT
CMD gunicorn --workers=4 --preload --bind 0.0.0.0:$PORT /Users/daewoongjun/Desktop/Programming/Python/DaewoongJun_Portfolio/amazon-rev-sentiment-analysis/amazon-review-sentiment-analysis/app/wsgi