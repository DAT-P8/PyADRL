from ..ngw.v1.ngw2d_pb2 import (
    MapSpec as GRPC_MapSpec,
    SquareMap as GRPC_SquareMap,
    ObjectSpec as GRPC_ObjectSpec,
    SquareObject as GRPC_SquareObject,
)
from typing import Self

class SquareObject:
    def to_dto(self) -> GRPC_SquareObject:
        return GRPC_SquareObject(x=self.x, y=self.y)

    @classmethod
    def from_dto(cls, square_object: GRPC_SquareObject) -> Self:
        return cls(square_object.x, square_object.y)

    def __init__(
        self,
        x: int,
        y: int,
    ) -> None:
        self.x = x
        self.y = y

class ObjectSpec:
    CASE_NAME = "object_oneof"
    SQUARE_OBJECT_CASE="square_object"

    def to_dto(self) -> GRPC_ObjectSpec:
        if self.square_object is not None:
            return GRPC_ObjectSpec(square_object=self.square_object.to_dto())
        else:
            raise ValueError("Did not find an object that was not none")

    @classmethod
    def from_dto(cls, object_spec: GRPC_ObjectSpec) -> Self:
        match object_spec.WhichOneof(ObjectSpec.CASE_NAME):
            case ObjectSpec.SQUARE_OBJECT_CASE:
                return cls(square_object=SquareObject.from_dto(object_spec.square_object))
            case e:
                raise ValueError(f"Did not recognize WhichOneof case: {e}")

    def __init__(
        self,
        square_object: SquareObject | None = None
    ) -> None:
        self.square_object = square_object

class SquareMap:
    def to_dto(self) -> GRPC_SquareMap:
        return GRPC_SquareMap(
            width=self.width,
            height=self.height,
            target_x=self.target_x,
            target_y=self.target_y,
            objects=[x.to_dto() for x in self.objects]
        )

    @classmethod
    def from_dto(cls, square_map: GRPC_SquareMap) -> Self:
        return cls(
            square_map.width,
            square_map.height,
            square_map.target_x,
            square_map.target_y,
            [ObjectSpec.from_dto(x) for x in square_map.objects],
        )

    def __init__(
        self,
        width: int,
        height: int,
        target_x: int,
        target_y: int,
        objects: list[ObjectSpec],
    ) -> None:
        self.width = width
        self.height = height
        self.target_x = target_x
        self.target_y = target_y
        self.objects = objects

class MapSpec:
    CASE_NAME = "map_oneof"
    SQUARE_MAP_CASE="square_map"

    def to_dto(self) -> GRPC_MapSpec:
        if self.square_map is not None:
            return GRPC_MapSpec(square_map=self.square_map.to_dto())
        else:
            raise ValueError("Did not find any not-null maps")

    @classmethod
    def from_dto(cls, map_spec: GRPC_MapSpec) -> Self:
        match map_spec.WhichOneof(MapSpec.CASE_NAME):
            case MapSpec.SQUARE_MAP_CASE:
                return cls(square_map=SquareMap.from_dto(map_spec.square_map))
            case e:
                raise ValueError(f"Did not recognize map case: {e}")

    def __init__(
        self,
        square_map: SquareMap | None = None
    ) -> None:
        self.square_map = square_map
