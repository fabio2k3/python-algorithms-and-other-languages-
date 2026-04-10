from .linkedlist import DoublyLinkedList


class State:
    def __init__(self, tiles):
        self.tiles = [row[:] for row in tiles]
        self.prev = None

    def __repr__(self):
        s = ""
        for i in range(len(self.tiles)):
            for j in range(len(self.tiles[i])):
                s += "{} ".format(self.tiles[i][j])
            s += "\n"
        return s

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return self.tiles == other.tiles

    def __hash__(self):
        return hash(tuple(tuple(x) for x in self.tiles))

    def __lt__(self, other):
        return str(self) < str(other)

    def copy(self):
        tiles = []
        for i in range(len(self.tiles)):
            tiles.append([])
            for j in range(len(self.tiles[i])):
                tiles[i].append(self.tiles[i][j])   # <- append correcto
        return State(tiles)

    def is_goal(self):
        N = len(self.tiles)
        counter = 1

        for i in range(N):
            for j in range(N):
                if i == N - 1 and j == N - 1:
                    return self.tiles[i][j] == " "
                if self.tiles[i][j] != counter:
                    return False
                counter += 1

        return True

    def is_solvable(self):
        flat = []
        for row in self.tiles:
            for x in row:
                if x != " ":
                    flat.append(x)

        inversions = 0
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                if flat[i] > flat[j]:
                    inversions += 1

        return inversions % 2 == 0

    def get_neighbs(self):
        N = len(self.tiles)
        neighbs = []

        row = 0
        col = 0

        for i in range(N):
            for j in range(N):
                if self.tiles[i][j] == " ":
                    row = i
                    col = j
                    break

        for i, j in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
            if 0 <= i < N and 0 <= j < N:
                n = self.copy()
                n.tiles[row][col], n.tiles[i][j] = n.tiles[i][j], n.tiles[row][col]
                neighbs.append(n)

        return neighbs

    def reconstruct_path(self):
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.prev
        path.reverse()
        return path

    def solve(self):
        if self.is_goal():
            return [self]

        if not self.is_solvable():
            return None

        frontier = DoublyLinkedList()
        frontier.add_last(self)

        visited = set()
        visited.add(self)

        while len(frontier) > 0:
            current = frontier.remove_first()

            if current.is_goal():
                return current.reconstruct_path()

            for neighb in current.get_neighbs():
                if neighb not in visited:
                    neighb.prev = current
                    visited.add(neighb)
                    frontier.add_last(neighb)

        return None


if __name__ == "__main__":
    initial = [
        [1, 2, 3],
        [4, 5, 6],
        [7, " ", 8]
    ]

    s = State(initial)
    solution = s.solve()

    if solution is None:
        print("No hay solución.")
    else:
        print(f"Solución encontrada en {len(solution) - 1} movimientos:\n")
        for i, state in enumerate(solution):
            print(f"Paso {i}:")
            print(state)