from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GWAction(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNKNOWN: _ClassVar[GWAction]
    NOTHING: _ClassVar[GWAction]
    LEFT: _ClassVar[GWAction]
    RIGHT: _ClassVar[GWAction]
    UP: _ClassVar[GWAction]
    DOWN: _ClassVar[GWAction]

UNKNOWN: GWAction
NOTHING: GWAction
LEFT: GWAction
RIGHT: GWAction
UP: GWAction
DOWN: GWAction

class GWDroneState(_message.Message):
    __slots__ = ("id", "x", "y", "destroyed", "is_evader")
    ID_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    DESTROYED_FIELD_NUMBER: _ClassVar[int]
    IS_EVADER_FIELD_NUMBER: _ClassVar[int]
    id: int
    x: int
    y: int
    destroyed: bool
    is_evader: bool
    def __init__(
        self,
        id: _Optional[int] = ...,
        x: _Optional[int] = ...,
        y: _Optional[int] = ...,
        destroyed: bool = ...,
        is_evader: bool = ...,
    ) -> None: ...

class GWState(_message.Message):
    __slots__ = ("terminated", "drone_states")
    TERMINATED_FIELD_NUMBER: _ClassVar[int]
    DRONE_STATES_FIELD_NUMBER: _ClassVar[int]
    terminated: bool
    drone_states: _containers.RepeatedCompositeFieldContainer[GWDroneState]
    def __init__(
        self,
        terminated: bool = ...,
        drone_states: _Optional[_Iterable[_Union[GWDroneState, _Mapping]]] = ...,
    ) -> None: ...

class GWDroneAction(_message.Message):
    __slots__ = ("id", "action")
    ID_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    id: int
    action: GWAction
    def __init__(
        self, id: _Optional[int] = ..., action: _Optional[_Union[GWAction, str]] = ...
    ) -> None: ...

class GWActionRequest(_message.Message):
    __slots__ = ("id", "drone_actions")
    ID_FIELD_NUMBER: _ClassVar[int]
    DRONE_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    id: int
    drone_actions: _containers.RepeatedCompositeFieldContainer[GWDroneAction]
    def __init__(
        self,
        id: _Optional[int] = ...,
        drone_actions: _Optional[_Iterable[_Union[GWDroneAction, _Mapping]]] = ...,
    ) -> None: ...

class GWActionResponse(_message.Message):
    __slots__ = ("state", "error_message")
    STATE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    state: GWState
    error_message: str
    def __init__(
        self,
        state: _Optional[_Union[GWState, _Mapping]] = ...,
        error_message: _Optional[str] = ...,
    ) -> None: ...

class GWResetRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: int
    def __init__(self, id: _Optional[int] = ...) -> None: ...

class GWResetResponse(_message.Message):
    __slots__ = ("state",)
    STATE_FIELD_NUMBER: _ClassVar[int]
    state: GWState
    def __init__(self, state: _Optional[_Union[GWState, _Mapping]] = ...) -> None: ...

class GWCloseRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: int
    def __init__(self, id: _Optional[int] = ...) -> None: ...

class GWCloseResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GWNewRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GWNewResponse(_message.Message):
    __slots__ = ("id", "state")
    ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    id: int
    state: GWState
    def __init__(
        self,
        id: _Optional[int] = ...,
        state: _Optional[_Union[GWState, _Mapping]] = ...,
    ) -> None: ...
