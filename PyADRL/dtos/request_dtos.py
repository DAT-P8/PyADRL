from typing import Self
from PyADRL.dtos.ngw_dtos import State
from ..ngw.v1.ngw2d_pb2 import (
    StateResponse as GRPC_StateResponse
)

class StateResponse:
    CASE_NAME="state_or_error"
    STATE_CASE="state"
    ERROR_CASE="error_message"

    @classmethod
    def from_dto(cls, response: GRPC_StateResponse) -> Self:
        match response.WhichOneof(StateResponse.CASE_NAME):
            case StateResponse.STATE_CASE:
                return cls(state=State.from_dto(response.state))
            case StateResponse.ERROR_CASE:
                return cls(error_message=response.error_message)
            case e:
                raise ValueError(f"Did not recognize case name: {e}")

    def to_dto(self) -> GRPC_StateResponse:
        if self.error_message is not None:
            return GRPC_StateResponse(error_message=self.error_message)
        elif self.state is not None:
            return GRPC_StateResponse(state=self.state.to_dto())
        else:
            raise ValueError("Did not find any non-null members")

    def __init__(
        self,
        error_message: str | None = None,
        state: State | None = None,
    ) -> None:
        self.error_message = error_message
        self.state = state
