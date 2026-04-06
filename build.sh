#!/usr/bin/env bash

set -e

cd "$(dirname $0)"

python3 -m grpc_tools.protoc -Iprotos --python_out=PyADRL --pyi_out=PyADRL --grpc_python_out=PyADRL protos/ngw/v1/ngw2d.proto

#python3 -m grpc_tools.protoc -Iprotos --python_out=PyADRL --pyi_out=PyADRL --grpc_python_out=PyADRL protos/grid_world.proto
#python3 -m grpc_tools.protoc -Iprotos --python_out=PyADRL --pyi_out=PyADRL --grpc_python_out=PyADRL protos/TDF.proto

# Fix import so it gets ngw2d_pb2 from the correct place (. instead of ngw.v1)
sed -i '' 's/^import ngw2d_pb2/from . import ngw2d_pb2/' 'PyAdRL/ngw/v1/ngw2d_pb2_grpc.py'
#sed -i '' 's/^import grid_world_pb2/from . import grid_world_pb2/' PyADRL/grid_world_pb2_grpc.py
#sed -i '' 's/^import TDF_pb2/from . import TDF_pb2/' PyADRL/TDF_pb2_grpc.py

echo 'import grpc.experimental' > delme.txt
cat ./PyADRL/ngw/v1/ngw2d_pb2_grpc.py >> delme.txt
cat delme.txt > ./PyADRL/ngw/v1/ngw2d_pb2_grpc.py

# echo "import grpc.experimental" > delme.txt
# cat ./PyADRL/grid_world_pb2_grpc.py >> delme.txt
# cat delme.txt > ./PyADRL/grid_world_pb2_grpc.py

# echo "import grpc.experimental" > delme.txt
# cat ./PyADRL/TDF_pb2_grpc.py >> delme.txt
# cat delme.txt > ./PyADRL/TDF_pb2_grpc.py

rm delme.txt
