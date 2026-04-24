from grpc import (
    Channel
)

from PyADRL.dtos.map_dtos import (
    MapSpec
)
from PyADRL.dtos.ngw_dtos import (
    DroneAction,
    State
)
from PyADRL.dtos.request_dtos import StateResponse
from PyADRL.ngw.v1.ngw2d_pb2 import CloseRequest, CloseResponse, DoStepRequest, DoStepResponse, NewRequest, NewResponse, ResetRequest, ResetResponse
from ..ngw.v1.ngw2d_pb2_grpc import (
    SimulationServiceStub
)

class NGWClient:
    def __init__(self, channel: Channel):
        self.stub = SimulationServiceStub(channel)

    def DoStep(self, id: int, actions: list[DroneAction]) -> State:
        req = DoStepRequest(sim_id=id, drone_actions=[a.to_dto() for a in actions])
        response: DoStepResponse = self.stub.DoStep(req)
        parsed_response = StateResponse.from_dto(response.state_response)
        if parsed_response.error_message is not None:
            raise ValueError(f"Error response from DoStep: {parsed_response.error_message}")
        if parsed_response.state is None:
            raise ValueError(f"Received a None state")

        return parsed_response.state

    def New(self, map: MapSpec, evader_count: int, pursuer_count: int) -> State:
        req = NewRequest(map=map.to_dto(), evader_count=evader_count, pursuer_count=pursuer_count)
        res: NewResponse = self.stub.New(req)
        parsed_response = StateResponse.from_dto(res.state_response)
        if parsed_response.error_message is not None:
            raise ValueError(f"Error response from New: {parsed_response.error_message}")
        if parsed_response.state is None:
            raise ValueError(f"Received a None state")

        return parsed_response.state
    
    def Reset(self, id: int) -> State:
        req = ResetRequest(sim_id=id)
        res: ResetResponse = self.stub.Reset(req)
        parsed_response = StateResponse.from_dto(res.state_response)
        if parsed_response.error_message is not None:
            raise ValueError(f"Error response from New: {parsed_response.error_message}")
        if parsed_response.state is None:
            raise ValueError(f"Received a None state")

        return parsed_response.state
    
    def Close(self, id: int) -> None:
        req = CloseRequest(sim_id=id)
        res: CloseResponse = self.stub.Reset(req)
        if res.error_message and len(res.error_message) != 0:
            raise ValueError(f"Error response from New: {res.error_message}")
