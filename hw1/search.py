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
    print("start: ", start, " end: ", maze.waypoints[0])
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
    print("reached end")
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
    return []

def astar_corner(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    return []

def astar_multiple(maze):
    """
    Runs A star for part 4 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []

def fast(maze):
    """
    Runs suboptimal search algorithm for part 5.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []
