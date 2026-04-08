import grpc
from ..ngw.v1 import ngw2d_pb2, ngw2d_pb2_grpc
# from .. import grid_world_pb2
# from .. import grid_world_pb2_grpc


class GridWorldClient:
    def __init__(self, channel: grpc.Channel):
        self.stub = ngw2d_pb2_grpc.SimulationServiceStub(channel)

    def Reset(self, request: ngw2d_pb2.ResetRequest) -> ngw2d_pb2.ResetResponse:
        return self.stub.Reset(request)

    def DoStep(self, request: ngw2d_pb2.DroneAction) -> ngw2d_pb2.DroneAction:
        return self.stub.DoStep(request)

    def New(self, request: ngw2d_pb2.NewRequest) -> ngw2d_pb2.NewResponse:
        return self.stub.New(request)

    def Close(self, request: ngw2d_pb2.CloseRequest) -> ngw2d_pb2.CloseResponse:
        return self.stub.Close(request)
