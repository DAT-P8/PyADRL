from grpc import Channel

from typing import Generic, TypeVar, Iterable

from ..TDF_pb2_grpc import TDFSimulationStub
from ..TDF_pb2 import (
    TDFCloseRequest,
    TDFCloseResponse,
    TDFDoStepRequest,
    TDFDoStepResponse,
    TDFNewRequest,
    TDFNewResponse,
    TDFResetRequest,
    TDFResetResponse,
    TDFState,
    TDFDroneAction,
)


class TDDroneAction:
    def __init__(self, id: int, x_f: float, y_f: float, z_f: float) -> None:
        self.id = id
        self.x_f = x_f
        self.y_f = y_f
        self.z_f = z_f

    def to_dto(self) -> TDFDroneAction:
        return TDFDroneAction(id=self.id, x_f=self.x_f, y_f=self.y_f, z_f=self.z_f)


T = TypeVar("T")
R = TypeVar("R")


class Result(Generic[T, R]):
    def __init__(self, ok: T | None = None, notok: R | None = None) -> None:
        super().__init__()

        if ok is None and notok is None:
            raise Exception("Result type cannot have null for both ok and not ok")

        self.okv = ok
        self.notokv = notok

    def is_ok(self) -> bool:
        return self.okv is not None

    def ok(self) -> T:
        if self.okv is None:
            raise Exception("Result was not ok")

        return self.okv

    def err(self) -> R:
        if self.notokv is None:
            raise Exception("Err was not set")

        return self.notokv


class TDClient:
    def __init__(self, channel: Channel) -> None:
        self.stub = TDFSimulationStub(channel)

    def New(
        self,
        evader_count: int,
        pursuer_count: int,
        evader_dome_radius: float,
        pursuer_dome_radius: float,
        arena_dome_radius: float,
        drone_max_speed: float,
        seed: int,
    ) -> Result[TDFState, str]:
        request: TDFNewRequest = TDFNewRequest(
            evader_count=evader_count,
            pursuer_count=pursuer_count,
            evader_dome_radius=evader_dome_radius,
            pursuer_dome_radius=pursuer_dome_radius,
            arena_dome_radius=arena_dome_radius,
            drone_max_speed=drone_max_speed,
            seed=seed,
        )
        response: TDFNewResponse = self.stub.New(request)

        e = response.WhichOneof("error_case")
        if e == "state":
            return Result(ok=response.state, notok=None)
        elif e == "error_msg":
            return Result(ok=None, notok=response.error_msg)
        else:
            raise Exception(f"Did not recognize response case: {e}")

    def DoStep(
        self, id: int, actions: Iterable[TDDroneAction]
    ) -> Result[TDFState, str]:
        request: TDFDoStepRequest = TDFDoStepRequest(
            id=id, drone_actions=[a.to_dto() for a in actions]
        )
        response: TDFDoStepResponse = self.stub.DoStep(request)

        e = response.WhichOneof("error_case")
        if e == "state":
            return Result(ok=response.state, notok=None)
        elif e == "error_msg":
            return Result(ok=None, notok=response.error_msg)
        else:
            raise Exception(f"Did not recognize response case: {e}")

    def Close(self, id: int) -> Result[None, str]:
        request: TDFCloseRequest = TDFCloseRequest(id=id)
        response: TDFCloseResponse = self.stub.Close(request)
        if response.HasField("error_msg"):
            return Result(notok=response.error_msg)
        else:
            return Result(ok=None)

    def Reset(self, id: int) -> Result[TDFState, str]:
        request: TDFResetRequest = TDFResetRequest(id=id)
        response: TDFResetResponse = self.stub.Reset(request)

        e = response.WhichOneof("error_case")
        if e == "state":
            return Result(ok=response.state)
        elif e == "error_msg":
            return Result(notok=response.error_msg)
        else:
            raise Exception(f"Did not recognize response case: {e}")
