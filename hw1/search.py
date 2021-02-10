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
    #print(out)
    return out

def astar_multiple(maze):
    """
    Runs A star for part 4 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    out = [] # path outputted
    queue = [] # prio queue based off g(n) + h(n) and contains location and waypoints hit
    heapq.heapify(queue)
    heuristic = {} # dictionary of prim's algo with manhattan distance, (x,y),(waypoints) : lowest manhattan distance cost to all remaining waypoints
    parents = {} # (x,y),(waypoints remaining) : ((parent x,y),(waypoints reamining), pathcost)
    waypoint_cost = {} # dictionary containing cost from one waypoint to the other waypoints (x,y),(x,y) : manhattan distance cost
    waypoints = [] #
    endgame = []
    prev_mst = {} # remaining tuples as input: mst heuristic cost
    # init lists in dict of waypoint_cost
    for w in maze.waypoints:
        waypoint_cost[w] = []
        waypoints.append(0)
        endgame.append(1)
    # calculating manhattan distances between waypoints
    for i in range(0, len(maze.waypoints)-1):
        for j in range(i+1, len(maze.waypoints)):
            distance = abs(maze.waypoints[i][0] - maze.waypoints[j][0]) + abs(maze.waypoints[i][1] - maze.waypoints[j][1])
            waypoint_cost[maze.waypoints[i]].append((distance, maze.waypoints[j]))
            waypoint_cost[maze.waypoints[j]].append((distance, maze.waypoints[i]))
    waypoints = tuple(waypoints) #(0,0,...,0)
    endgame = tuple(endgame) # (1,1,...,1)
    # init start state
    start = (maze.start, waypoints)
    parents[start] = ((-1,-1), waypoints), 0
    heapq.heappush(queue, (0, start))
    curr = start
    # time to start searching
    while len(queue) > 0 and curr[1] != endgame:
        curr = heapq.heappop(queue)[1]
        #print(curr, curr[1])
        # current location matches an undiscovered waypoint
        if curr[0] in maze.waypoints:# and curr[1][maze.waypoints.index(curr[0])] == 0:
            #print("found")
            index = maze.waypoints.index(curr[0])
            waypoints = list(curr[1])
            waypoints[index] = 1
            checkpoint = (curr[0], tuple(waypoints))
            #print(checkpoint)
            parents[checkpoint] = parents[curr]
            curr = checkpoint
        for n in maze.neighbors(curr[0][0], curr[0][1]):
            node = (n, curr[1])
            path_cost = parents[curr][1] + 1
            if node not in parents:
                parents[node] = curr,path_cost
                # calculate prim's
                cost = 0
                remaining = []
                mst = []
                for i in range(len(curr[1])):
                    if i == 0:
                        distance = abs(n[0]-maze.waypoints[i][0]) + abs(n[1]-maze.waypoints[i][1])
                        mst.append((distance,(maze.waypoints[i][0], maze.waypoints[i][1])))
                        remaining.append(0)
                    else:
                        remaining.append(1)
                og = tuple(remaining)
                temp = mst[0][0]
                if og in prev_mst:
                    cost = cost + mst[0][0] + prev_mst[og]
                else:
                    while tuple(remaining) != endgame:
                        mst.sort()
                        index = maze.waypoints.index(mst[0][1])
                        if remaining[index] == 0:
                            cost = cost + mst[0][0]
                            mst.insert(0, waypoint_cost[mst[0][1]])
                            remaining[index] = 1
                            del mst[0]
                        else:
                            del mst[0]
                    prev_mst[og] = cost - temp
                cost = cost + parents[node][1]
                heapq.heappush(queue, (cost, node))
                heuristic[node] = cost
            elif path_cost < parents[node][1]:
                old_cost = parents[node][1]
                heuristic[node] = heuristic[node] - old_cost + path_cost
                parents[node] = curr,path_cost
                heapq.heappush(queue, (heuristic[node], node))
    #print("Done")
    while curr != start:
        #print(curr)
        out.insert(0, curr[0]
        parent = parents[curr][0]
        curr = parent
    out.insert(0, maze.start)
    return out

def fast(maze):
    """
    Runs suboptimal search algorithm for part 5.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []
