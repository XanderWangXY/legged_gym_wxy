import importlib

terrain_registry = dict(
    MyTerrain= "legged_gym.utils.my_terrain.terrain:MyTerrain",
    BarrierTrack= "legged_gym.utils.my_terrain.barrier_track:BarrierTrack",
    TerrainPerlin= "legged_gym.utils.my_terrain.perlin:TerrainPerlin",
)

def get_terrain_cls(terrain_cls):
    entry_point = terrain_registry[terrain_cls]
    module, class_name = entry_point.rsplit(":", 1)
    module = importlib.import_module(module)
    return getattr(module, class_name)
