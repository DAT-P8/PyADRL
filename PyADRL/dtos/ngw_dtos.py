from typing import Self
from ..ngw.v1.ngw2d_pb2 import (
    Action as GRPC_Action,
    DroneAction as GRPC_DroneAction,
    DroneState as GRPC_DroneState,
    State as GRPC_State,
    Event as GRPC_Event,
    DroneObjectCollisionEvent as GRPC_DroneObjectCollisionEvent,
    TargetReachedEvent as GRPC_TargetReachedEvent,
    CollisionEvent as GRPC_CollisionEvent,
    OutOfBoundsEvent as GRPC_OutOfBoundsEvent,
    PursuerEnteredTargetEvent as GRPC_PursuerEnteredTargetEvent,
)


class Action:
    ACTION_UNKNOWN_UNSPECIFIED = 0
    ACTION_NOTHING = 1
    ACTION_LEFT = 2
    ACTION_LEFT_UP = 3
    ACTION_UP = 4
    ACTION_RIGHT_UP = 5
    ACTION_RIGHT = 6
    ACTION_RIGHT_DOWN = 7
    ACTION_DOWN = 8
    ACTION_LEFT_DOWN = 9

    def __init__(self, val: int) -> None:
        self.val = val
        pass

    def to_dto(self) -> GRPC_Action:
        match self.val:
            case self.ACTION_UNKNOWN_UNSPECIFIED:
                return GRPC_Action.ACTION_UNKNOWN_UNSPECIFIED

            case self.ACTION_NOTHING:
                return GRPC_Action.ACTION_NOTHING

            case self.ACTION_LEFT:
                return GRPC_Action.ACTION_LEFT

            case self.ACTION_LEFT_UP:
                return GRPC_Action.ACTION_LEFT_UP

            case self.ACTION_UP:
                return GRPC_Action.ACTION_UP

            case self.ACTION_RIGHT_UP:
                return GRPC_Action.ACTION_RIGHT_UP

            case self.ACTION_RIGHT:
                return GRPC_Action.ACTION_RIGHT

            case self.ACTION_RIGHT_DOWN:
                return GRPC_Action.ACTION_RIGHT_DOWN

            case self.ACTION_DOWN:
                return GRPC_Action.ACTION_DOWN

            case self.ACTION_LEFT_DOWN:
                return GRPC_Action.ACTION_LEFT_DOWN

            case e:
                raise ValueError(f"Did not recognize action: {e}")

    @classmethod
    def from_dto(cls, action: GRPC_Action) -> Self:
        return cls(Action.action_to_val(action))

    @staticmethod
    def action_to_val(action: GRPC_Action) -> int:
        match action:
            case GRPC_Action.ACTION_UNKNOWN_UNSPECIFIED:
                return Action.ACTION_UNKNOWN_UNSPECIFIED
            case GRPC_Action.ACTION_NOTHING:
                return Action.ACTION_NOTHING
            case GRPC_Action.ACTION_LEFT:
                return Action.ACTION_LEFT
            case GRPC_Action.ACTION_LEFT_UP:
                return Action.ACTION_LEFT_UP
            case GRPC_Action.ACTION_UP:
                return Action.ACTION_UP
            case GRPC_Action.ACTION_RIGHT_UP:
                return Action.ACTION_RIGHT_UP
            case GRPC_Action.ACTION_RIGHT:
                return Action.ACTION_RIGHT
            case GRPC_Action.ACTION_RIGHT_DOWN:
                return Action.ACTION_RIGHT_DOWN
            case GRPC_Action.ACTION_DOWN:
                return Action.ACTION_DOWN
            case GRPC_Action.ACTION_LEFT_DOWN:
                return Action.ACTION_LEFT_DOWN
            case e:
                raise ValueError(f"Did not recognize action: {e}")


class DroneAction:
    def to_dto(self) -> GRPC_DroneAction:
        return GRPC_DroneAction(
            id=self.id, action=self.action.to_dto(), velocity=self.velocity
        )

    @classmethod
    def from_dto(cls, drone_action: GRPC_DroneAction) -> Self:
        return cls(
            drone_action.id, drone_action.velocity, Action.from_dto(drone_action.action)
        )

    def __init__(self, id: int, velocity: int, action: Action) -> None:
        self.id = id
        self.velocity = velocity
        self.action = action


class DroneState:
    def to_dto(self) -> GRPC_DroneState:
        return GRPC_DroneState(
            id=self.id,
            x=self.x,
            y=self.y,
            destroyed=self.destroyed,
            is_evader=self.is_evader,
        )

    @classmethod
    def from_dto(cls, drone_state: GRPC_DroneState) -> Self:
        return cls(
            drone_state.id,
            drone_state.x,
            drone_state.y,
            drone_state.destroyed,
            drone_state.is_evader,
        )

    def __init__(
        self,
        id: int,
        x: int,
        y: int,
        destroyed: bool = False,
        is_evader: bool = False,
    ):
        self.id = id
        self.x = x
        self.y = y
        self.destroyed = destroyed
        self.is_evader = is_evader


class DroneObjectCollisionEvent:
    def to_dto(self) -> GRPC_DroneObjectCollisionEvent:
        return GRPC_DroneObjectCollisionEvent(drone_ids=self.drone_ids)

    @classmethod
    def from_dto(cls, event: GRPC_DroneObjectCollisionEvent):
        ids = [x for x in event.drone_ids]
        return cls(ids)

    def __init__(self, drone_ids: list[int]) -> None:
        self.drone_ids: list[int] = drone_ids


class CollisionEvent:
    def to_dto(self) -> GRPC_CollisionEvent:
        return GRPC_CollisionEvent(drone_ids=self.drone_ids)

    @classmethod
    def from_dto(cls, event: GRPC_CollisionEvent):
        ids = [x for x in event.drone_ids]
        return cls(ids)

    def __init__(self, drone_ids: list[int]) -> None:
        self.drone_ids: list[int] = drone_ids


class TargetReachedEvent:
    def to_dto(self) -> GRPC_TargetReachedEvent:
        return GRPC_TargetReachedEvent(drone_ids=self.drone_ids)

    @classmethod
    def from_dto(cls, event: GRPC_TargetReachedEvent):
        ids = [x for x in event.drone_ids]
        return cls(ids)

    def __init__(self, drone_ids: list[int]) -> None:
        self.drone_ids: list[int] = drone_ids


class OutOfBoundsEvent:
    def to_dto(self) -> GRPC_OutOfBoundsEvent:
        return GRPC_OutOfBoundsEvent(drone_ids=self.drone_ids)

    @classmethod
    def from_dto(cls, event: GRPC_OutOfBoundsEvent):
        ids = [x for x in event.drone_ids]
        return cls(ids)

    def __init__(self, drone_ids: list[int]) -> None:
        self.drone_ids: list[int] = drone_ids


class PursuerEnteredTargetEvent:
    def to_dto(self) -> GRPC_PursuerEnteredTargetEvent:
        return GRPC_PursuerEnteredTargetEvent(drone_ids=self.drone_ids)

    @classmethod
    def from_dto(cls, event: GRPC_PursuerEnteredTargetEvent):
        ids = [x for x in event.drone_ids]
        return cls(ids)

    def __init__(self, drone_ids: list[int]) -> None:
        self.drone_ids: list[int] = drone_ids


class Event:
    CASE_NAME = "event_oneof"
    COLLISION_EVENT_CASE = "collision_event"
    TARGET_REACHED_EVENT_CASE = "target_reached_event"
    OUT_OF_BOUNDS_EVENT_CASE = "out_of_bounds_event"
    PURSUER_ENTERED_TARGET_EVENT_CASE = "pursuer_entered_target_event"
    DRONE_OBJECT_COLLISION_EVENT_CASE = "drone_object_collision_event"

    def to_dto(self) -> GRPC_Event:
        if self.collision_event is not None:
            return GRPC_Event(collision_event=self.collision_event.to_dto())
        elif self.target_reached_event is not None:
            return GRPC_Event(target_reached_event=self.target_reached_event.to_dto())
        elif self.out_of_bounds_event is not None:
            return GRPC_Event(out_of_bounds_event=self.out_of_bounds_event.to_dto())
        elif self.pursuer_entered_target_event is not None:
            return GRPC_Event(
                pursuer_entered_target_event=self.pursuer_entered_target_event.to_dto()
            )
        elif self.drone_object_collision_event is not None:
            return GRPC_Event(
                drone_object_collision_event=self.drone_object_collision_event.to_dto()
            )
        else:
            raise ValueError("Did not find a not null event")

    @classmethod
    def from_dto(cls, event: GRPC_Event) -> Self:
        match event.WhichOneof(Event.CASE_NAME):
            case Event.TARGET_REACHED_EVENT_CASE:
                return cls(
                    target_reached_event=TargetReachedEvent.from_dto(
                        event.target_reached_event
                    )
                )
            case Event.PURSUER_ENTERED_TARGET_EVENT_CASE:
                return cls(
                    pursuer_entered_target_event=PursuerEnteredTargetEvent.from_dto(
                        event.pursuer_entered_target_event
                    )
                )
            case Event.COLLISION_EVENT_CASE:
                return cls(
                    collision_event=CollisionEvent.from_dto(event.collision_event)
                )
            case Event.OUT_OF_BOUNDS_EVENT_CASE:
                return cls(
                    out_of_bounds_event=OutOfBoundsEvent.from_dto(
                        event.out_of_bounds_event
                    )
                )
            case Event.DRONE_OBJECT_COLLISION_EVENT_CASE:
                return cls(
                    drone_object_collision_event=DroneObjectCollisionEvent.from_dto(
                        event.drone_object_collision_event
                    )
                )
            case e:
                raise ValueError(f"Did not find a matching event type for {e}")

    def __init__(
        self,
        collision_event: CollisionEvent | None = None,
        target_reached_event: TargetReachedEvent | None = None,
        out_of_bounds_event: OutOfBoundsEvent | None = None,
        pursuer_entered_target_event: PursuerEnteredTargetEvent | None = None,
        drone_object_collision_event: DroneObjectCollisionEvent | None = None,
    ) -> None:
        self.collision_event = collision_event
        self.target_reached_event = target_reached_event
        self.out_of_bounds_event = out_of_bounds_event
        self.pursuer_entered_target_event = pursuer_entered_target_event
        self.drone_object_collision_event = drone_object_collision_event


class State:
    def to_dto(self) -> GRPC_State:
        return GRPC_State(
            sim_id=self.sim_id,
            terminated=self.terminated,
            drone_states=[x.to_dto() for x in self.drone_states],
            events=[x.to_dto() for x in self.events],
        )

    @classmethod
    def from_dto(cls, state: GRPC_State) -> Self:
        return cls(
            state.sim_id,
            state.terminated,
            [DroneState.from_dto(x) for x in state.drone_states],
            [Event.from_dto(x) for x in state.events],
        )

    def __init__(
        self,
        sim_id: int,
        terminated: bool,
        drone_states: list[DroneState],
        events: list[Event],
    ):
        self.sim_id = sim_id
        self.terminated = terminated
        self.drone_states = drone_states
        self.events = events
