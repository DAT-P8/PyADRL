#!/usr/bin/env bash

set -e

cd "$(dirname $0)"

python -m grpc_tools.protoc -Iprotos --python_out=PyADRL --pyi_out=PyADRL --grpc_python_out=PyADRL protos/ngw/v1/ngw2d.proto

sed -i.bak 's/from ngw\.v1 import/from . import/' 'PyADRL/ngw/v1/ngw2d_pb2_grpc.py'
rm 'PyADRL/ngw/v1/ngw2d_pb2_grpc.py.bak'

echo 'import grpc.experimental' > delme.txt
cat ./PyADRL/ngw/v1/ngw2d_pb2_grpc.py >> delme.txt
cat delme.txt > ./PyADRL/ngw/v1/ngw2d_pb2_grpc.py

rm delme.txt
