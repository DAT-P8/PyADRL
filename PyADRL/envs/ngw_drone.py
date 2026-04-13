class NGW_Drone:
    def __init__(self, id: int, x: int, y: int, is_evader: bool):
        self.id: int = id
        self.x = x
        self.y = y
        self.name = f"evader_{id}" if is_evader else f"pursuer_{id}"
        self.is_evader = is_evader
        self.destroyed: bool = False
