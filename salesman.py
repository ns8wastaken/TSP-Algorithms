import random

Vec2 = tuple[int, int]

class Salesman:
    """
    NearestNeighbor() - Starts at a point and keeps adding the nearest neighbor to the path until it makes a loop
    NearestNeighborParallel() - Starts at a point and keeps track of 2 separate paths that keep adding their nearest neighbors to the paths until they merge
    AntColony() - Starts at a random point but chosen paths are random so it also attributes rewards to shorter paths to increase the chance of a short path being chosen
    TwoOptOptimization() - Check if swapping any 2 edges gives a shorter path
    ThreeOptOptimization() - Check if swapping any 3 edges gives a shorter path

    points: list of 2D coordinates
    distances: a matrix of distances

    [AntColony]
    iterations: number of times to simulate the ants (more is better but there is a falloff)

    [TwoOptOptimization - ThreeOptOptimization]
    defaultPoints: if the path was modified then the points wont be in the default order, the default order is needed
    """

    @staticmethod
    def NearestNeighbor(points: list[Vec2], distances: list[list[int]]) -> tuple[list[Vec2], float]:
        point_to_index = {p:i for i, p in enumerate(points)}
        path = [ points[0] ]
        length = 0.0

        for _ in range(len(points) - 1):
            minDist = float("inf")
            minPoint = path[-1]
            minPointIndex = point_to_index[path[-1]]

            for j, p2 in enumerate(points):
                if (p2 not in path) and (minPoint != p2):
                    dist = distances[minPointIndex][j]
                    if dist <= minDist:
                        minPoint = p2
                        minDist = dist

            path.append(minPoint)
            length += minDist

        path.append(path[0])
        length += distances[0][len(path) - 2]
        return (path, length)

    @staticmethod
    def NearestNeighborParallel(points: list[Vec2], distances: list[list[int]]) -> tuple[list[Vec2], float]:
        point_to_index = {p:i for i, p in enumerate(points)}
        path1 = [ points[0] ]
        path2 = [ points[0] ]
        length = 0.0

        for _ in range(len(points) // 2):
            minDist = float("inf")
            minPoint = path1[-1]
            minPointIndex = point_to_index[path1[-1]]

            for j, p2 in enumerate(points):
                if (p2 not in path1) and (p2 not in path2):
                    dist = distances[minPointIndex][j]
                    if dist <= minDist:
                        minPoint = p2
                        minDist = dist

            path1.append(minPoint)
            length += minDist

            minDist = float("inf")
            minPoint = path2[-1]
            minPointIndex = point_to_index[path2[-1]]

            for j, p2 in enumerate(points):
                if (p2 not in path1) and (p2 not in path2):
                    dist = distances[minPointIndex][j]
                    if dist <= minDist:
                        minPoint = p2
                        minDist = dist

            path2.append(minPoint)
            length += minDist

        path2.reverse()
        length += distances[points.index(path1[-1])][points.index(path2[0])]
        return (path1 + path2, length)

    @staticmethod
    def AntColony(points: list[Vec2], distances: list[list[int]], iterations: int) -> tuple[list[Vec2], float]:
        point_to_index = {p:i for i, p in enumerate(points)}

        rewards = [[1.0 for _ in range(len(distances))] for _ in range(len(distances))]

        best_path = [ points[0] ]
        best_length = float("inf")

        for _ in range(iterations):
            path = [ random.choice(points) ]
            length = 0.0

            for _ in range(len(points) - 1):
                pts = []
                weights = []
                lastPointIndex = point_to_index[path[-1]]

                probs_sum = sum([(distances[lastPointIndex][i] ** -1) * (rewards[lastPointIndex][i] ** 6) for i, p in enumerate(points) if p not in path])

                for j, p2 in enumerate(points):
                    if p2 not in path:
                        pts.append(p2)
                        dist = distances[lastPointIndex][j]
                        weights.append( ((dist ** -1) * (rewards[lastPointIndex][j] ** 6)) / probs_sum )

                next_point = random.choices(pts, weights)[0]
                path.append(next_point)
                length += distances[lastPointIndex][point_to_index[next_point]]

                rewards[lastPointIndex][point_to_index[next_point]] += (length ** -1) * 10000
                rewards[point_to_index[next_point]][lastPointIndex] += (length ** -1) * 10000

            path.append(path[0])

            for i, r in enumerate(rewards):
                for j in range(len(r)):
                    rewards[i][j] *= 0.98

            if length < best_length:
                best_length = length
                best_path = path

        return (best_path, best_length)

    @staticmethod
    def TwoOptOptimize(defaultPoints: list[Vec2], path: list[Vec2], current_dist: float, distances: list[list[int]]) -> tuple[list[Vec2], float]:
        point_to_index = {p:i for i, p in enumerate(defaultPoints)}

        improvement = True
        best_dist = current_dist

        n = len(path)
        while improvement:
            improvement = False

            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    new_path = path[:i] + path[i:j + 1][::-1] + path[j + 1:] # 2-Opt Swap
                    dist = sum([ distances[ point_to_index[new_path[x]] ][ point_to_index[new_path[x + 1]] ] for x in range(len(new_path) - 1) ])

                    if dist < best_dist:
                        path = new_path
                        best_dist = dist
                        improvement = True

            if improvement:
                print(f"2-Opt improved distance: {best_dist}")

        return (path, best_dist)

    @staticmethod
    def ThreeOptOptimize(defaultPoints: list[Vec2], path: list[Vec2], current_dist: float, distances: list[list[int]]) -> tuple[list[Vec2], float]:
        point_to_index = {p:i for i, p in enumerate(defaultPoints)}

        def ThreeOptSwap(path: list[Vec2], a: int, c: int, e: int):
            b, d, f = a + 1, c + 1, e + 1

            paths = []

            # 3-opt (a, d) (e, c) (b, f)
            paths.append(path[:a + 1] + path[d:e + 1] + path[c:b - 1:-1] + path[f:])
            # 3-opt (a, d) (e, b) (c, f)
            paths.append(path[:a + 1] + path[d:e + 1] + path[b:c + 1] + path[f:])
            # 3-opt (a, e) (d, b) (c, f)
            paths.append(path[:a + 1] + path[e:d - 1:-1] + path[b:c + 1] + path[f:])
            # 3-opt (a, c) (b, e) (d, f)
            paths.append(path[:a + 1] + path[c:b - 1:-1] + path[e:d - 1:-1] + path[f:])

            return paths

        best_distance = current_dist
        improvement = True

        n = len(path)
        while improvement:
            improvement = False

            for i in range(n - 5):
                for j in range(i + 2, n - 3):
                    for k in range(j + 2, n - 1):
                        new_paths = ThreeOptSwap(path, i, j, k)

                        for new_path in new_paths:
                            dist = sum([ distances[ point_to_index[new_path[x]] ][ point_to_index[new_path[x + 1]] ] for x in range(len(new_path) - 1) ])

                            if dist < best_distance:
                                path = new_path
                                best_distance = dist
                                improvement = True

            if improvement:
                print(f'3-Opt improved distance: {best_distance}')

        return (path, best_distance)
