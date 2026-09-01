# region Do not edit, your solution will be invalidated

from typing import Any, Set, List
from dataclasses import dataclass

import random
import time


@dataclass
class Position:
    x: int
    y: int

    def __eq__(self, value: object) -> bool:
        return self.x == value.x and self.y == value.y

    def __ne__(self, value: object) -> bool:
        return self.x != value.x or self.y != value.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))


@dataclass
class Sheep:
    position: Position
    points: int

    def __lt__(self, other) -> bool:
        return self.points < other.points


@dataclass
class Field:
    length: int
    walls: Set[Position]
    sheep: List[Sheep]
    rivals: List[Position]


# This function is called by the racing software, do not edit its signature
def move(field: Field, current_position: Position) -> Position:
    # endregion
    # region Editable code
    valid_moves = get_valid_neighbors(current_position, field)
    random_move = random.choice(valid_moves)
    # The algorithm is currently to move randomly. This is what you want to improve
    return random_move


def get_valid_neighbors(current_position: Position, field: Field) -> List[Position]:
    possible_neighbors = [
        Position(current_position.x, current_position.y - 1),
        Position(current_position.x, current_position.y + 1),
        Position(current_position.x + 1, current_position.y),
        Position(current_position.x - 1, current_position.y),
    ]
    valid_neighbors = []
    for neighbor in possible_neighbors:
        is_valid = (
            0 <= neighbor.x < field.length
            and 0 <= neighbor.y < field.length
            and neighbor not in field.walls
        )
        if is_valid:
            valid_neighbors.append(neighbor)
    return valid_neighbors
# endregion
#region Helper functions for testing and visualization
def field_from_string(map_string: str) -> Field:
    lines = map_string.strip().splitlines()
    length = int(lines[0])
    walls = set()
    sheep = []
    rivals = []
    for y, line in enumerate(lines[1:]):
        for x, ch in enumerate(line):
            if ch == 'D':
                walls.add(Position(x, y))
            elif ch == 'P':
                rivals.append(Position(x, y))
            elif ch.isdigit() and ch != '0':
                sheep.append(Sheep(Position(x, y), int(ch)))
    return Field(length, walls, sheep, rivals)

# This is purely for your conveniance. It will not be used in the competition. It is tested on a windows powershell, but you may have to adjust to your environment
def show_grid(field: Field, player: Position):
    grid = [[" " for _ in range(field.length)] for _ in range(field.length)]
    for sheep in field.sheep:
        grid[sheep.position.y][sheep.position.x] = str(sheep.points)
    for wall in field.walls:
        grid[wall.y][wall.x] = "X"

    grid[player.y][player.x] = "P"
    print("\033[H\033[J", end="")
    border_h = "+" + "-" * field.length + "+"
    print(border_h)
    for row in grid:
        print("|" + "".join(row) + "|")
    print(border_h)


def use_default_field():
    # Example grid. The competition grid will be different
    return Field(
        length=50,
        walls={
            Position(x=38, y=23),
            Position(x=42, y=39),
            Position(x=7, y=26),
            Position(x=18, y=26),
            Position(x=22, y=26),
            Position(x=29, y=41),
            Position(x=40, y=41),
            Position(x=9, y=17),
            Position(x=9, y=26),
            Position(x=24, y=8),
            Position(x=41, y=24),
            Position(x=9, y=35),
            Position(x=18, y=10),
            Position(x=21, y=9),
            Position(x=26, y=23),
            Position(x=13, y=26),
            Position(x=24, y=26),
            Position(x=30, y=39),
            Position(x=43, y=39),
            Position(x=11, y=44),
            Position(x=40, y=34),
            Position(x=47, y=27),
            Position(x=40, y=43),
            Position(x=28, y=23),
            Position(x=22, y=37),
            Position(x=31, y=40),
            Position(x=9, y=19),
            Position(x=41, y=26),
            Position(x=32, y=23),
            Position(x=1, y=26),
            Position(x=31, y=15),
            Position(x=11, y=46),
            Position(x=34, y=23),
            Position(x=40, y=36),
            Position(x=30, y=13),
            Position(x=9, y=12),
            Position(x=3, y=26),
            Position(x=14, y=26),
            Position(x=43, y=7),
            Position(x=40, y=45),
            Position(x=9, y=21),
            Position(x=41, y=28),
            Position(x=42, y=27),
            Position(x=27, y=38),
            Position(x=19, y=34),
            Position(x=16, y=26),
            Position(x=31, y=8),
            Position(x=45, y=7),
            Position(x=46, y=27),
            Position(x=25, y=38),
            Position(x=9, y=42),
            Position(x=11, y=48),
            Position(x=12, y=13),
            Position(x=17, y=9),
            Position(x=20, y=26),
            Position(x=29, y=38),
            Position(x=20, y=35),
            Position(x=40, y=38),
            Position(x=9, y=14),
            Position(x=40, y=47),
            Position(x=41, y=21),
            Position(x=32, y=18),
            Position(x=41, y=30),
            Position(x=43, y=27),
            Position(x=35, y=23),
            Position(x=41, y=39),
            Position(x=27, y=40),
            Position(x=31, y=1),
            Position(x=20, y=10),
            Position(x=45, y=9),
            Position(x=31, y=10),
            Position(x=25, y=40),
            Position(x=31, y=19),
            Position(x=40, y=31),
            Position(x=37, y=23),
            Position(x=40, y=40),
            Position(x=21, y=36),
            Position(x=9, y=16),
            Position(x=41, y=14),
            Position(x=43, y=11),
            Position(x=44, y=10),
            Position(x=23, y=42),
            Position(x=41, y=23),
            Position(x=30, y=20),
            Position(x=45, y=39),
            Position(x=10, y=26),
            Position(x=31, y=3),
            Position(x=9, y=37),
            Position(x=31, y=12),
            Position(x=22, y=9),
            Position(x=11, y=43),
            Position(x=12, y=8),
            Position(x=49, y=27),
            Position(x=14, y=14),
            Position(x=40, y=33),
            Position(x=12, y=26),
            Position(x=23, y=26),
            Position(x=40, y=42),
            Position(x=43, y=13),
            Position(x=13, y=9),
            Position(x=41, y=25),
            Position(x=30, y=22),
            Position(x=25, y=26),
            Position(x=31, y=5),
            Position(x=9, y=39),
            Position(x=31, y=14),
            Position(x=11, y=45),
            Position(x=31, y=23),
            Position(x=48, y=39),
            Position(x=40, y=35),
            Position(x=20, y=32),
            Position(x=21, y=31),
            Position(x=40, y=44),
            Position(x=41, y=9),
            Position(x=42, y=8),
            Position(x=10, y=12),
            Position(x=33, y=23),
            Position(x=9, y=23),
            Position(x=16, y=16),
            Position(x=19, y=33),
            Position(x=2, y=26),
            Position(x=40, y=10),
            Position(x=31, y=7),
            Position(x=9, y=41),
            Position(x=31, y=16),
            Position(x=11, y=47),
            Position(x=14, y=9),
            Position(x=6, y=26),
            Position(x=24, y=41),
            Position(x=26, y=38),
            Position(x=40, y=37),
            Position(x=4, y=26),
            Position(x=45, y=27),
            Position(x=41, y=20),
            Position(x=11, y=13),
            Position(x=16, y=9),
            Position(x=8, y=26),
            Position(x=9, y=25),
            Position(x=19, y=26),
            Position(x=28, y=38),
            Position(x=22, y=43),
            Position(x=39, y=13),
            Position(x=40, y=12),
            Position(x=31, y=9),
            Position(x=11, y=49),
            Position(x=21, y=26),
            Position(x=26, y=40),
            Position(x=19, y=10),
            Position(x=10, y=7),
            Position(x=42, y=12),
            Position(x=27, y=23),
            Position(x=9, y=18),
            Position(x=31, y=2),
            Position(x=9, y=36),
            Position(x=15, y=15),
            Position(x=40, y=14),
            Position(x=31, y=11),
            Position(x=48, y=27),
            Position(x=29, y=23),
            Position(x=46, y=39),
            Position(x=44, y=39),
            Position(x=40, y=32),
            Position(x=42, y=14),
            Position(x=9, y=11),
            Position(x=11, y=8),
            Position(x=30, y=21),
            Position(x=9, y=20),
            Position(x=0, y=26),
            Position(x=11, y=26),
            Position(x=41, y=27),
            Position(x=31, y=4),
            Position(x=9, y=38),
            Position(x=23, y=9),
            Position(x=15, y=26),
            Position(x=24, y=38),
            Position(x=9, y=13),
            Position(x=40, y=46),
            Position(x=17, y=26),
            Position(x=30, y=23),
            Position(x=47, y=39),
            Position(x=9, y=22),
            Position(x=23, y=39),
            Position(x=32, y=17),
            Position(x=13, y=13),
            Position(x=41, y=29),
            Position(x=31, y=6),
            Position(x=9, y=40),
            Position(x=45, y=8),
            Position(x=36, y=23),
            Position(x=9, y=6),
            Position(x=40, y=39),
            Position(x=9, y=15),
            Position(x=5, y=26),
            Position(x=21, y=44),
            Position(x=9, y=24),
            Position(x=41, y=22),
            Position(x=40, y=11),
            Position(x=44, y=27),
        },
        sheep=[
            Sheep(position=Position(x=12, y=0), points=4),
            Sheep(position=Position(x=23, y=0), points=8),
            Sheep(position=Position(x=45, y=0), points=7),
            Sheep(position=Position(x=7, y=1), points=3),
            Sheep(position=Position(x=34, y=1), points=3),
            Sheep(position=Position(x=0, y=2), points=4),
            Sheep(position=Position(x=15, y=2), points=5),
            Sheep(position=Position(x=8, y=3), points=9),
            Sheep(position=Position(x=30, y=3), points=6),
            Sheep(position=Position(x=45, y=3), points=7),
            Sheep(position=Position(x=23, y=4), points=3),
            Sheep(position=Position(x=15, y=5), points=4),
            Sheep(position=Position(x=35, y=5), points=4),
            Sheep(position=Position(x=13, y=6), points=6),
            Sheep(position=Position(x=29, y=6), points=5),
            Sheep(position=Position(x=8, y=7), points=5),
            Sheep(position=Position(x=20, y=8), points=7),
            Sheep(position=Position(x=23, y=8), points=9),
            Sheep(position=Position(x=44, y=8), points=2),
            Sheep(position=Position(x=35, y=9), points=4),
            Sheep(position=Position(x=42, y=10), points=3),
            Sheep(position=Position(x=0, y=11), points=4),
            Sheep(position=Position(x=7, y=11), points=3),
            Sheep(position=Position(x=27, y=11), points=6),
            Sheep(position=Position(x=15, y=12), points=5),
            Sheep(position=Position(x=31, y=13), points=5),
            Sheep(position=Position(x=41, y=13), points=9),
            Sheep(position=Position(x=13, y=14), points=6),
            Sheep(position=Position(x=37, y=15), points=4),
            Sheep(position=Position(x=8, y=16), points=9),
            Sheep(position=Position(x=11, y=17), points=2),
            Sheep(position=Position(x=31, y=17), points=6),
            Sheep(position=Position(x=43, y=17), points=6),
            Sheep(position=Position(x=16, y=18), points=2),
            Sheep(position=Position(x=31, y=18), points=8),
            Sheep(position=Position(x=37, y=18), points=8),
            Sheep(position=Position(x=42, y=19), points=3),
            Sheep(position=Position(x=7, y=21), points=3),
            Sheep(position=Position(x=31, y=21), points=3),
            Sheep(position=Position(x=16, y=23), points=2),
            Sheep(position=Position(x=8, y=24), points=3),
            Sheep(position=Position(x=17, y=24), points=8),
            Sheep(position=Position(x=28, y=24), points=3),
            Sheep(position=Position(x=43, y=24), points=2),
            Sheep(position=Position(x=42, y=26), points=5),
            Sheep(position=Position(x=44, y=26), points=5),
            Sheep(position=Position(x=9, y=27), points=4),
            Sheep(position=Position(x=28, y=27), points=3),
            Sheep(position=Position(x=18, y=28), points=8),
            Sheep(position=Position(x=42, y=28), points=9),
            Sheep(position=Position(x=14, y=30), points=4),
            Sheep(position=Position(x=30, y=30), points=5),
            Sheep(position=Position(x=38, y=30), points=2),
            Sheep(position=Position(x=23, y=32), points=4),
            Sheep(position=Position(x=5, y=33), points=4),
            Sheep(position=Position(x=14, y=33), points=9),
            Sheep(position=Position(x=41, y=33), points=7),
            Sheep(position=Position(x=42, y=33), points=3),
            Sheep(position=Position(x=21, y=34), points=8),
            Sheep(position=Position(x=27, y=35), points=4),
            Sheep(position=Position(x=6, y=36), points=4),
            Sheep(position=Position(x=42, y=36), points=3),
            Sheep(position=Position(x=38, y=38), points=3),
            Sheep(position=Position(x=25, y=39), points=9),
            Sheep(position=Position(x=5, y=40), points=5),
            Sheep(position=Position(x=13, y=40), points=2),
            Sheep(position=Position(x=44, y=40), points=5),
            Sheep(position=Position(x=46, y=47), points=7),
            Sheep(position=Position(x=12, y=48), points=2),
            Sheep(position=Position(x=22, y=48), points=1),
            Sheep(position=Position(x=4, y=49), points=8),
            Sheep(position=Position(x=22, y=49), points=5),
            Sheep(position=Position(x=49, y=49), points=4),
        ],
        rivals=[Position(x=22, y=25)],
    )

if __name__ == "__main__":
    # Example: Load field from a file
    # with open("map_file.txt", "r") as f:
    #     field = field_from_string(f.read())

    field = use_default_field()

    # You are the only one in this field, but in the competition you will have rivals!
    current_position = field.rivals[0]
    move_number = 0
    score = 0
    auto_until = 0
    while len(field.sheep) > 0:
        current_position = move(field, current_position)
        move_number += 1
        collected = [s for s in field.sheep if current_position == s.position]
        score += sum(s.points for s in collected)
        field.sheep = [s for s in field.sheep if current_position != s.position]
        show_grid(field, current_position)
        print(f"Move: {move_number} | Score: {score} | Sheep remaining: {len(field.sheep)} | Position: ({current_position.x}, {current_position.y})")
        if move_number < auto_until:
            time.sleep(0.1)
        else:
            cmd = input("Enter to step, or a number to auto-advance to that move: ").strip()
            if cmd.isdigit():
                auto_until = int(cmd)

# endregion
