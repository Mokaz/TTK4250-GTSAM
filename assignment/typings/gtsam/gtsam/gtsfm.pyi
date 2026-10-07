"""
gtsfm submodule
"""
from __future__ import annotations
import collections.abc
import gtsam.gtsam
import numpy
import numpy.typing
import typing
__all__: list[str] = ['Keypoints', 'tracksFromPairwiseMatches']
class Keypoints:
    def __init__(self, coordinates: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 2]"]) -> None:
        ...
    @property
    def coordinates(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 2]"]:
        ...
    @coordinates.setter
    def coordinates(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 2]"]) -> None:
        ...
def tracksFromPairwiseMatches(matches_dict: collections.abc.Mapping[gtsam.gtsam.IndexPair, typing.Annotated[numpy.typing.ArrayLike, numpy.int32, "[m, 2]"]], keypoints_list: collections.abc.Sequence[Keypoints], verbose: bool = False) -> list[gtsam.gtsam.SfmTrack2d]:
    ...
