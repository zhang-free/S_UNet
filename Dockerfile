FROM neubiaswg5/ml-keras-base

ADD wrapper.py /app/wrapper.py
ADD unet.py /app/unet.py

ENTRYPOINT ["python", "/app/wrapper.py"]