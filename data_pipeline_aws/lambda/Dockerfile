FROM public.ecr.aws/lambda/python:3.11

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["lambda_function.lambda_handler"]
