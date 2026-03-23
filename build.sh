#!/usr/bin/env bash

set -e

cd "$(dirname $0)"

python -m grpc_tools.protoc -Iprotos --python_out=PyADRL --pyi_out=PyADRL --grpc_python_out=PyADRL protos/grid_world.proto
python -m grpc_tools.protoc -Iprotos --python_out=PyADRL --pyi_out=PyADRL --grpc_python_out=PyADRL protos/TDF.proto

sed -i 's/^import grid_world_pb2/from . import grid_world_pb2/' PyADRL/grid_world_pb2_grpc.py
sed -i 's/^import TDF_pb2/from . import TDF_pb2/' PyADRL/TDF_pb2_grpc.py

echo "import grpc.experimental" > delme.txt
cat ./PyADRL/grid_world_pb2_grpc.py >> delme.txt
cat delme.txt > ./PyADRL/grid_world_pb2_grpc.py

echo "import grpc.experimental" > delme.txt
cat ./PyADRL/TDF_pb2_grpc.py >> delme.txt
cat delme.txt > ./PyADRL/TDF_pb2_grpc.py

rm delme.txt
