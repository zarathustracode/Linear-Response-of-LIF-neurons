FROM ubuntu:focal

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y libboost-all-dev python3-dev git build-essential cmake g++ gdb python3-dbg python3-pip

COPY ./lifAPI /app/lifAPI
COPY ./requirements.txt /app

WORKDIR /app/lifAPI

RUN cmake .
RUN make

WORKDIR /app

RUN pip3 install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "lifAPI.main:app", "--host=0.0.0.0", "--reload"]
