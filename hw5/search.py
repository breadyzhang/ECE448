import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]

###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return (value, [ move ], { encode(*move): {} })
    else:
        return (evaluate(board), [], {})

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.
def minimax(side, board, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    value = math.inf if side == True else -math.inf
    moveList = []
    moveTree = {}
    bestMove = []
    # create minimax tree move:(next move, heuristic)
    # bubble up from leaves
    # base case
    if depth == 1:
        # find all possible moves at board state
        for move in generateMoves(side,board,flags):
            moveTree[encode(move[0],move[1],move[2])] = {}
            newside,newboard,newflags = makeMove(side,board,move[0],move[1],flags,move[2])
            # check if move is optimal
            if side == True and evaluate(newboard) < value: # min player which means wants lowest value
                value = evaluate(newboard)
                bestMove = move
            elif side == False and evaluate(newboard) > value:
                value = evaluate(newboard)
                bestMove = move
        moveList.append(bestMove)
        return value,moveList,moveTree

    for move in generateMoves(side,board,flags):
        moveTree[encode(move[0],move[1],move[2])] = {}
        newside, newboard, newflags = makeMove(side,board,move[0],move[1],flags,move[2])
        minmax = minimax(newside,newboard,newflags,depth-1)
        moveTree[encode(move[0],move[1],move[2])] = minmax[2]
        moves = minmax[2]
        if side == True and minmax[0] < value: # min player which means wants lowest value
            value = minmax[0]
            bestMove = move
            moveList = minmax[1]
        elif side == False and minmax[0] > value:
            value = minmax[0]
            bestMove = move
            moveList = minmax[1]
    moveList.insert(0,bestMove)
    return value,moveList,moveTree

def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    value = 0
    moveList = []
    moveTree = {}
    if depth == 0:
        return evaluate(board),moveList,moveTree
    if side == False: # max/white
        value = -math.inf
        for move in generateMoves(side,board,flags):
            newside,newboard,newflags = makeMove(side,board,move[0],move[1],flags,move[2])
            alphabet = alphabeta(newside,newboard,newflags,depth-1,alpha,beta)
            moveTree[encode(move[0],move[1],move[2])] = alphabet[2]
            if alphabet[0] > value:
                value = alphabet[0]
                alphabet[1].insert(0,move)
                moveList = alphabet[1].copy()
            alpha = max(alpha,value)
            if alpha >= beta: # prune check
                break

        # print("depth: ",depth)
        # print(value)
        # print(moveList)
        # # print(moveTree)
        # print("list len: ",len(moveList))
        return value,moveList,moveTree
    else: # min/black
        value = math.inf
        for move in generateMoves(side,board,flags):
            newside,newboard,newflags = makeMove(side,board,move[0],move[1],flags,move[2])
            alphabet = alphabeta(newside,newboard,newflags,depth-1,alpha,beta)
            moveTree[encode(move[0],move[1],move[2])] = alphabet[2]
            if alphabet[0] < value:
                value = alphabet[0]
                alphabet[1].insert(0,move)
                moveList = alphabet[1].copy()
            beta = min(beta,value)
            if beta <= alpha: # prune check
                break

        # print("depth: ", depth)
        # print(value)
        # print(moveTree)
        # print(moveList)
        return value,moveList,moveTree


def stochastic(side, board, flags, depth, breadth, chooser):
    '''
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (value, moveList, moveTree)
      value (float): average board value of the paths for the best-scoring move
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    value = 0
    moveList = []
    moveTree = {}
    initialMoves = []
    if side == True: # black moves
        for move in generateMoves(side,board,flags):
            newside,newboard,newflags = makeMove(side,board,move[0],move[1],flags,move[2])
            moveTree[encode(move[0],move[1],move[2])] = {}
            # print("\nmove: ",move,"\nvalue: ",evaluate(newboard))
            # find all possible next moves
            random = stochastic_helper(newside,newboard,newflags,depth-1,breadth,chooser,depth)
            # print("next value: ",random[0])
            moveTree[encode(move[0],move[1],move[2])] = random[2]
            random[1].insert(0,move)
            initialMoves.append((random[0],random[1]))
        value,moveList = min(initialMoves)
    else: # white moves
        for move in generateMoves(side,board,flags):
            newside,newboard,newflags = makeMove(side,board,move[0],move[1],flags,move[2])
            moveTree[encode(move[0],move[1],move[2])] = {}
            # find all possible next moves
            random = stochastic_helper(newside,newboard,newflags,depth-1,breadth,chooser,depth)
            moveTree[encode(move[0],move[1],move[2])] = random[2]
            random[1].insert(0,move)
            initialMoves.append((random[0],random[1]))
        value,moveList = max(initialMoves)
    # print(initialMoves)
    # print(min(initialMoves))
    # print(value)
    # print(moveList)
    # print(moveTree)
    return value,moveList,moveTree

def stochastic_helper(side,board,flags,level,breadth,chooser,depth):
    value = 0
    moveList = []
    moveTree = {}
    moves = []
    initialMoves = []
    # base case
    if level == 0:
        return evaluate(board),moveList,moveTree
    # create list of possible moves for chooser
    for move in generateMoves(side,board,flags):
        moves.append(move)
    if level == depth-1:
        for i in range(breadth):
            move = chooser(moves)
            newside,newboard,newflags = makeMove(side,board,move[0],move[1],flags,move[2])
            random = stochastic_helper(newside,newboard,newflags,level-1,breadth,chooser,depth)
            moveTree[encode(move[0],move[1],move[2])] = random[2]
            random[1].insert(0,move)
            initialMoves.append((random[0],random[1]))
            # value = random[0] + evaluate(board)
            # moveList = random[1].copy()
            # moveList.insert(0,move)
        # print(initialMoves)
        for next in initialMoves:
            value += next[0]
        value = value/breadth
        if side == True:
            moveList = min(initialMoves)[1]
        else:
            moveList = max(initialMoves)[1]
        # print(initialMoves)
    else:
        move = chooser(moves)
        newside,newboard,newflags = makeMove(side,board,move[0],move[1],flags,move[2])
        random = stochastic_helper(newside,newboard,newflags,level-1,breadth,chooser,depth)
        moveTree[encode(move[0],move[1],move[2])] = random[2]
        value = random[0]
        moveList = random[1].copy()
        moveList.insert(0,move)
    return value,moveList,moveTree
