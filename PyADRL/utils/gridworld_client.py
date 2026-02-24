
import grpc
import grid_world_pb2, grid_world_pb2_grpc

class GridWorldClient:
    def __init__(self, channel: grpc.Channel):
        self.stub = grid_world_pb2_grpc.GWSimulationStub(channel)

    def Reset(self, request: grid_world_pb2.GWResetRequest) -> grid_world_pb2.GWResetResponse:
        return self.stub.Reset(request)

    def DoStep(self, request: grid_world_pb2.GWActionRequest) -> grid_world_pb2.GWActionResponse:
        return self.stub.DoStep(request)

    def New(self, request: grid_world_pb2.GWNewRequest) -> grid_world_pb2.GWNewResponse:
        return self.stub.New(request)

    def Close(self, request: grid_world_pb2.GWCloseRequest) -> grid_world_pb2.GWCloseResponse:
        return self.stub.Close(request)


    