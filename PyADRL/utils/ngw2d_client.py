from grpc import Channel

from PyADRL.dtos.map_dtos import MapSpec
from PyADRL.dtos.ngw_dtos import DroneAction, State
from PyADRL.dtos.request_dtos import StateResponse
from PyADRL.ngw.v1.ngw2d_pb2 import (
    CloseRequest,
    CloseResponse,
    DoStepRequest,
    DoStepResponse,
    NewRequest,
    NewResponse,
    ResetRequest,
    ResetResponse,
)
from ..ngw.v1.ngw2d_pb2_grpc import SimulationServiceStub


class NGWClient:
    def __init__(self, channel: Channel, rpc_timeout: float | None = 5.0):
        """RPC client for NGW simulation. rpc_timeout is seconds passed to each RPC; None means no timeout."""
        self.stub = SimulationServiceStub(channel)
        self.rpc_timeout = rpc_timeout

    def DoStep(self, id: int, actions: list[DroneAction]) -> State:
        req = DoStepRequest(sim_id=id, drone_actions=[a.to_dto() for a in actions])
        # Pass rpc timeout so a hung server doesn't block training forever
        response: DoStepResponse = (
            self.stub.DoStep(req, timeout=self.rpc_timeout)
            if self.rpc_timeout is not None
            else self.stub.DoStep(req)
        )
        parsed_response = StateResponse.from_dto(response.state_response)
        if parsed_response.error_message is not None:
            raise ValueError(
                f"Error response from DoStep: {parsed_response.error_message}"
            )
        if parsed_response.state is None:
            raise ValueError("Received a None state")

        return parsed_response.state

    def New(self, map: MapSpec, evader_count: int, pursuer_count: int) -> State:
        req = NewRequest(
            map=map.to_dto(), evader_count=evader_count, pursuer_count=pursuer_count
        )
        res: NewResponse = (
            self.stub.New(req, timeout=self.rpc_timeout)
            if self.rpc_timeout is not None
            else self.stub.New(req)
        )
        parsed_response = StateResponse.from_dto(res.state_response)
        if parsed_response.error_message is not None:
            raise ValueError(
                f"Error response from New: {parsed_response.error_message}"
            )
        if parsed_response.state is None:
            raise ValueError("Received a None state")

        return parsed_response.state

    def Reset(self, id: int) -> State:
        req = ResetRequest(sim_id=id)
        res: ResetResponse = (
            self.stub.Reset(req, timeout=self.rpc_timeout)
            if self.rpc_timeout is not None
            else self.stub.Reset(req)
        )
        parsed_response = StateResponse.from_dto(res.state_response)
        if parsed_response.error_message is not None:
            raise ValueError(
                f"Error response from New: {parsed_response.error_message}"
            )
        if parsed_response.state is None:
            raise ValueError("Received a None state")

        return parsed_response.state

    def Close(self, id: int) -> None:
        req = CloseRequest(sim_id=id)
        if self.rpc_timeout is not None:
            _res: CloseResponse = self.stub.Close(req, timeout=self.rpc_timeout)
        else:
            _res: CloseResponse = self.stub.Close(req)
        if _res.HasField("error_message"):
            raise ValueError(f"Error response from Close: {_res.error_message}")
