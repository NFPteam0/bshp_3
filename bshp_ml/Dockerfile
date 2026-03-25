FROM python:3.12.4
 
LABEL authors="dark_h14"

WORKDIR /app
COPY ./requirements.txt .
EXPOSE 8061
RUN python3 -m pip install --root-user-action=ignore --upgrade pip && python3 -m  pip install --root-user-action=ignore  --trusted-host pypi.org -r requirements.txt

ENV PYTHONPATH=/app

# COPY ./app .

ENTRYPOINT ["uvicorn", "main_app:app", "--host", "0.0.0.0", "--port", "8061", "--reload"]