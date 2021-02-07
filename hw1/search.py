# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)
import heapq

def bfs(maze):
    # queue [] for points to visit in maze
    # point []:
        # point[0] = parent tuple
        # point[1] = current coord
        # start point[2:] = -1,-1
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    out = []
    queue = []
    parents = {}
    visited = {}
    start = (maze.start[0], maze.start[1])
    parents[start] = (-1, -1)
    queue.append(start)
    i = 0
    #print("start: ", start, " end: ", maze.waypoints[0])
    while(queue[i] != maze.waypoints[0]):
        #print("current point: ", queue[i])
        visited[queue[i]] = 1
        for n in maze.neighbors(queue[i][0], queue[i][1]):
            # if n in visited:
            #     print("already visited")
            if n not in visited:
                #print("parent: ", queue[i], "child: ", n)
                parents[n] = queue[i]
                queue.append(n)
                visited[n] = 1
        i = i + 1
    #print("reached end")
    curr = queue[i]
    while(curr != (-1,-1)):
        #print(curr)
        out.insert(0, curr)
        curr = parents[curr]
    return out

def astar_single(maze):
    # heapq
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # heapq: (distance, coords)
    out = []
    queue = []
    heapq.heapify(queue)
    parents = {}
    visited = {}
    goal = (maze.waypoints[0][0], maze.waypoints[0][1])
    start = (maze.start[0], maze.start[1])
    parents[start] = (-1,-1)
    curr = start
    visited[start] = 1
    while curr not in maze.waypoints:
        #print("curr: ", curr)
        for n in maze.neighbors(curr[0], curr[1]):
            if n not in visited:
                #print("neighbor: ", n)
                parents[n] = curr
                distance = abs(goal[0]-n[0]) + abs(goal[1]-n[1]) + visited[curr]
                #print(distance)
                heapq.heappush(queue, (distance, n))
                visited[n] = 1 + visited[curr]
        curr = heapq.heappop(queue)[1]
    while(curr != (-1,-1)):
        out.insert(0,curr)
        curr = parents[curr]
    return out

def astar_corner(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    out = []
    queue = [] # queue: distances to waypoints, [waypoints not yet hit], current location
    parents = {} # current location, (points not yet hit) : parent location, cost from start
    visited = {} # current location, (points not yet hit) : heuristic (manhattan distance from all remaining waypoints + path)
    heapq.heapify(queue)
    start = ((maze.start[0], maze.start[1]), (1,1,1,1))
    curr = start
    #print(curr)
    visited[start] = 0
    parents[start] = ((-1,-1), 0)
    while curr[1] != (0,0,0,0):
        # updates tuple of waypoints hit, set to 0
        if curr[0] in maze.waypoints:
            #print("found")
            index = maze.waypoints.index(curr[0])
            waypoints = list(curr[1])
            waypoints[index] = 0
            checkpoint = (curr[0], tuple(waypoints))
            #print(checkpoint)
            parents[checkpoint] = parents[curr]
            curr = checkpoint
        for n in maze.neighbors(curr[0][0],curr[0][1]):
            node = (n, curr[1])
            # calculate heuristic: manhattan distance to remaining waypoints + cost to get to current location
            cost = parents[curr][1]+1
            if node not in parents or cost < parents[node][1]:
                parents[node] = (curr, cost)
                for i in range(len(maze.waypoints)):
                    cost = cost + (abs(n[0]-maze.waypoints[i][0]) + abs(n[1]-maze.waypoints[i][1]))*curr[1][i]
                heapq.heappush(queue, (cost, node))
        #print(queue)
        curr = heapq.heappop(queue)[1]

    # insert path into output
    #print("adding to path", curr)
    while curr != start:
        parent = parents[curr]
        out.insert(0, curr[0])
        curr = parent[0]
        # print(curr)
        # print("Path: ", out)
    out.insert(0, maze.start)
    del out[-1]
    print(out)
    return out

def astar_multiple(maze):
    """
    Runs A star for part 4 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    distances = []
    for i in range(len(maze.waypoints)-1):
        for j in range(i+1, len(maze.waypoints)):
            distance = abs(maze.waypoints[i][0] - maze.waypoints[j][0]) + abs(maze.waypoints[i][1]-maze.waypoints[j][1])
            distances.append((distance, (maze.waypoints[i], maze.waypoints[j])))
    distances.sort()
    checked = []
    next = {}
    for i in distances:
        if i[1][0] not in checked or i[1][1] not in checked:
            checked.append(i[1][0])
            checked.append(i[1][1])
            next[i[1][0]] = i[1][1]
    return []

def fast(maze):
    """
    Runs suboptimal search algorithm for part 5.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []
