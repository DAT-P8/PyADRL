#!/usr/bin/env bash

set -e

cd "$(dirname $0)"

python -m grpc_tools.protoc -Iprotos --python_out=PyADRL --pyi_out=PyADRL --grpc_python_out=PyADRL protos/grid_world.proto

sed -i 's/^import grid_world_pb2/from . import grid_world_pb2/' PyADRL/grid_world_pb2_grpc.py
