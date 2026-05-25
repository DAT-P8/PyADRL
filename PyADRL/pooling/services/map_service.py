from logging import Logger
import json

from PyADRL.utils import paths


class MapObject:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


class Map:
    def __init__(
        self,
        width: int,
        height: int,
        target_x: int,
        target_y: int,
        objects: list[MapObject],
    ) -> None:
        self.width = width
        self.height = height
        self.target_x = target_x
        self.target_y = target_y
        self.objects = objects


class MapService:
    def __init__(self, logger: Logger) -> None:
        self.logger = logger
        pass

    def get_from_name(self, name: str) -> Map | None:
        self.logger.debug("getting map")

        try:
            maps_dir = paths.get_env_maps_dir()
            for file in maps_dir.iterdir():
                fname = file.name
                if "." in fname:
                    fname = fname.split(".")[0]

                if file.is_dir() or fname != name:
                    continue

                with open(file, "r") as f:
                    raw_object = json.load(f)

                width: int = int(raw_object["width"])
                height: int = int(raw_object["height"])
                target_x: int = int(raw_object["target_x"])
                target_y: int = int(raw_object["target_y"])

                objects = [
                    MapObject(x=int(o["x"]), y=int(o["y"]))
                    for o in list(raw_object["objects"])
                ]

                return Map(width, height, target_x, target_y, objects)

        except Exception as e:
            self.logger.error(f"error trying to read map: {e}")
            return None

        return None
