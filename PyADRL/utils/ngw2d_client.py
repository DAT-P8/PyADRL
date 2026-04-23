import grpc
from ..ngw.v1 import ngw2d_pb2, ngw2d_pb2_grpc


class EventTypes:
    CollisionEvent = "collision_event"
    TargetReachedEvent = "target_reached_event"
    OutOfBoundsEvent = "out_of_bounds_event"
    CaptureEvent = "capture_event"
    PursuerEnteredTargetEvent = "pursuer_entered_target_event"
    DroneObjectCollisionEvent = "drone_object_collision_event"


class DroneState:
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


class State:
    def __init__(
        self,
        sim_id: int | None = None,
        terminated: bool = False,
        drone_states: list[DroneState] = [],
        events: dict[EventTypes, list[list[int]]] = {},
        objects: list[ngw2d_pb2.ObjectSpec] = [],
    ):
        self.sim_id = sim_id
        self.terminated = terminated
        self.drone_states = drone_states
        self.events = events
        self.objects = objects


class NGWClient:
    def __init__(self, channel: grpc.Channel):
        self.stub = ngw2d_pb2_grpc.SimulationServiceStub(channel)

    def Reset(self, request: ngw2d_pb2.ResetRequest) -> State:
        reset_response = self.stub.Reset(request).state_response
        if reset_response.WhichOneof("state_or_error") == "state":
            return self.ReadState(reset_response.state)
        else:
            raise ValueError(
                f"Reset request responded with an error: {reset_response.error_message}"
            )

    def DoStep(self, request: ngw2d_pb2.DoStepRequest) -> State:
        do_step_response = self.stub.DoStep(request).state_response
        if do_step_response.WhichOneof("state_or_error") == "state":
            return self.ReadState(do_step_response.state)
        else:
            raise ValueError(
                f"DoStep request responded with an error: {do_step_response.error_message}"
            )

    def New(self, request: ngw2d_pb2.NewRequest) -> State:
        new_response = self.stub.New(request).state_response
        if new_response.WhichOneof("state_or_error") == "state":
            return self.ReadState(new_response.state)
        else:
            raise ValueError(
                f"New request responded with an error: {new_response.error_message}"
            )

    def Close(self, request: ngw2d_pb2.CloseRequest) -> None:
        close_response = self.stub.Close(request)
        if close_response.HasField("error_message"):
            raise ValueError(
                f"Close request responded with an error: {close_response.error_message}"
            )

    def ReadState(self, state: ngw2d_pb2.State) -> State:
        sim_id = state.sim_id
        terminated = state.terminated
        drone_states = [
            DroneState(ds.id, ds.x, ds.y, ds.destroyed, ds.is_evader)
            for ds in state.drone_states
        ]
        objects = list(state.objects)

        # Used to determine if a collision was a capture
        ids = {
            "evaders": [d.id for d in drone_states if d.is_evader],
            "pursuers": [d.id for d in drone_states if not d.is_evader],
        }

        events = {}
        for e in state.events:
            drone_ids = []
            event_type = None
            match e.WhichOneof("event_oneof"):
                case EventTypes.TargetReachedEvent as target_reached:
                    drone_ids = e.target_reached_event.drone_ids
                    event_type = target_reached
                case EventTypes.OutOfBoundsEvent as out_of_bounds:
                    drone_ids = e.out_of_bounds_event.drone_ids
                    event_type = out_of_bounds
                case EventTypes.CollisionEvent:
                    drone_ids = e.collision_event.drone_ids
                    # Determine if it was a collision or a capture
                    has_evader = any(d in ids["evaders"] for d in drone_ids)
                    has_pursuer = any(d in ids["pursuers"] for d in drone_ids)
                    event_type = (
                        EventTypes.CaptureEvent
                        if has_evader and has_pursuer
                        else EventTypes.CollisionEvent
                    )
                case EventTypes.PursuerEnteredTargetEvent as pusuer_in_target:
                    drone_ids = e.pursuer_entered_target_event.drone_ids
                    event_type = pusuer_in_target
                case EventTypes.DroneObjectCollisionEvent as drone_object_collision:
                    drone_ids = drone_object_collision.drone_ids
                    event_type = drone_object_collision
                case _ as unkown_event:
                    raise ValueError(f"Recieved unkown event type {unkown_event}")
            if event_type not in events.keys():
                events[event_type] = []
            events[event_type].append(drone_ids)
        return State(sim_id, terminated, drone_states, events, objects)
