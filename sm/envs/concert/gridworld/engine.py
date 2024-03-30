from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np

import time

from .items import ItemBase, ItemKind


class GridEngine:
    def __init__(self, shape: Tuple[int, int], rng: np.random.Generator = None) -> None:
        self.rng = rng or np.random.default_rng()
        self.shape = shape
        self.grid: Dict[Tuple[int, int], Set[ItemBase]] = defaultdict(set)
        self.grid_locs = np.stack(
            np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0])), -1
        )

    def reset(self):
        self.grid.clear()

    def add(self, items: List[ItemBase]) -> None:
        """Add a set of items to the grid world."""
        items_with_loc = [i for i in items if i.loc[0] > -1]
        items_without_locs = [i for i in items if i.loc[0] == -1]
        self._add(items_with_loc)
        self._add_randomly(items_without_locs)

    def _add(self, items: List[ItemBase]) -> None:
        for i in items:
            assert i.engine is None
            i.engine = self
            #if  a list of locations are passed or juts one location
            
            if any(isinstance(loc, np.ndarray) for loc in list(i.loc)):
                
                for loc_ in i.loc:
                    self.grid[tuple(loc_)].add(i)
            else:
                self.grid[tuple(i.loc)].add(i)

            
            # if len(i.loc)>1:
            #     for loc in (i.loc):
            #         self.grid[tuple(loc)].add(i)
            # else:
            


    def _add_randomly(self, items: List[ItemBase]) -> None:
        free_locs = self.passable_locs()  
        if len(free_locs) < len(items):
            raise ValueError("Failed to distribute items")
        c = self.rng.choice(len(free_locs), size=len(items), replace=False)
        for item, loc in zip(items, free_locs[c]):
            item.loc[:] = loc
        self._add(items)

    def is_passable(
        self, locs: np.ndarray, exclude_items: List[ItemBase] = None
    ) -> np.ndarray:
        """Test passable state of given locations"""
        locs = np.atleast_2d(locs)
        if exclude_items is None:
            exclude_items = []
        exset = set(exclude_items)
        p = []
        for loc in locs:
            s = self.grid[tuple(loc)]
            p.append(len(s) == 0 or all([i.passable for i in (s - exset)]))
        return np.array(p, dtype=bool)

    def passable_map(self) -> np.ndarray:
        return self.is_passable(self.grid_locs.reshape(-1, 2)).reshape(*self.shape)

    def passable_locs(self) -> np.ndarray:
        m = self.passable_map()
        ids = np.stack(np.where(m)[::-1], -1)
        return ids

    def items(
        self, locs: np.ndarray = None, order: List[ItemKind] = None
    ) -> List[ItemBase]:
        """Returns a list of all items optionally ordered by type."""
        if locs is None:
            items = list(set().union(*list(self.grid.values())))
        else:
            locs = np.atleast_2d(locs)
            
            for loc_ in locs:
                if isinstance(loc_[0], np.ndarray):
                    items = list(set().union(*[self.grid[tuple(loc)] for loc in loc_]))
                else:
                    items = list(set().union(*[self.grid[tuple(loc_)]]))
       
            
        if order is not None:
            items = sorted(items, key=lambda i: order.index(i.kind))
        return items

    def check_move(
        self,
        items: List[ItemBase],
        newlocs: np.ndarray,
        ignore_self_collisions: bool = True,
    ) -> bool:
        """Test move for potential problems.

        A move for a group of objects may only be successful if
         - no object moves out of bound
         - all objects move to unoccupied locations
        """

        # Out of bounds
        if not np.logical_and(
            newlocs >= 0, newlocs < np.array(self.shape).reshape(1, 2)
        ).all():
            return False

        # Collisions
        exclude = items if ignore_self_collisions else None
        if not all(self.is_passable(newlocs, exclude_items=exclude)):
            return False

        return True

    def offset_locs(self, items: List[ItemBase], off: np.ndarray) -> np.ndarray:
        """Computes the new locations for group of items moved by a common offset."""
        locs = np.stack([o.loc for o in items], 0)
        newlocs = locs + off
        return newlocs



     
        return locs

    def move(
        self, items: List[ItemBase], newlocs: np.ndarray, test: bool = True
    ) -> bool:
        """Moves a list of grid objects to new locations

        A move for a group of objects may only be successful if
         - no object moves out of bound
         - all objects move to unoccupied locations

        If any of the above conditions is violated, no object moves.
        Self-collisions resulting between the group of objects are
        correctly handled.

        Returns
        -------
        success: boolean
            Whether the move was successful or not
        """
        newlocs = np.atleast_2d(newlocs)
        if test and not self.check_move(items, newlocs):
            return False




        # Update
        for i, nloc in zip(items, newlocs):

            if ((isinstance(i.loc[0], np.ndarray))):

                
                for grid_num, grid_loc in enumerate(i.loc):
                    oldloc = tuple(grid_loc)

                    self.grid[oldloc].remove(i)
                    if len(self.grid[oldloc]) == 0:
                        del self.grid[oldloc]
                    self.grid[tuple(nloc[grid_num])].add(i)
            else:
                oldloc = tuple(i.loc)
                self.grid[oldloc].remove(i)
                if len(self.grid[oldloc]) == 0:
                    del self.grid[oldloc]
                self.grid[tuple(nloc)].add(i)
                i.loc[:] = nloc
        return True


if __name__ == "__main__":
    from .items import MoveAgent, GoalItem, ObjectItem, WallItem
    from .renderer import render

    shape = (8, 8)
    e = GridEngine(shape)
    w = WallItem.create_border(shape)
    e.add(w)

    @property
    def free_locations(self) -> np.ndarray:
        omap = self.occupancy_maps[0]
        ids = np.stack(np.where(omap == 0), -1)
        return ids

    agents = [MoveAgent((-1, -1)) for _ in range(2)]
    objs = [ObjectItem((-1, -1)) for _ in range(2)]
    goals = [GoalItem((-1, -1)) for _ in range(2)]
    e.add_randomly(agents + objs + goals)

    # free_locs = e.passable_locs()
    # c = np.random.choice(len(free_locs), size=6, replace=False)
    # agents = [AgentItem(free_locs[i]) for i in c[:2]]
    # objs = [ObjectItem(free_locs[i]) for i in c[2:4]]
    # goals = [GoalItem(free_locs[i]) for i in c[4:6]]
    # e.add(agents + objs + goals)

    img = render(e)
    import matplotlib.pyplot as plt

    plt.imshow(img, origin="upper")
    plt.show()
