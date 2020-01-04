# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import sys
from game import Directions

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Searches the deepest nodes in the search tree first.

    returns a list of actions that reaches the goal.
    """
    
    startState = problem.getStartState()
    # Set used to keep track of already explored nodes
    been = set()
    # Currently traversed path
    curPath = []
    fringe = util.Stack()
    # Store tuples of state and path used to reach that state
    fringe.push((startState, curPath))
    result = []
    while not fringe.isEmpty():
        # While the fringe is not empty, get the next tuple
        curState, curPath = fringe.pop()
        if problem.isGoalState(curState):
            # Check if state is the goal state.
            # If it is, store the current path and break the loop
            result = curPath
            break
        # Add state in the set so that we don't reach it twice
        been.add(curState)
        # Get the neighbour states of the current state
        successors = problem.getSuccessors(curState)
        for succ in successors:
            # Check whether we have been in this state before
            if succ[0] not in been:
                curPath.append(succ[1])
                # Save tuple of this neighbour state and path to reach it
                fringe.push((succ[0], list(curPath)))
                curPath.pop()
    return result


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.

    returns a list of actions that reaches the goal.
    """

    startState = problem.getStartState()
    # Set used to keep track of already explored nodes
    been = set()
    # Currently traversed path
    curPath = []
    fringe = util.Queue()
    # Store tuples of state and path used to reach that state
    fringe.push((startState, curPath))
    result = []
    while not fringe.isEmpty():
        # While the fringe is not empty, get the next tuple
        curState, curPath = fringe.pop()
        # Check whether we have been in this state before
        if curState in been: continue
        # Add state in the set so that we don't reach it twice
        been.add(curState)
        if problem.isGoalState(curState):
            # Check if state is the goal state.
            # If it is, store the current path and break the loop
            result = curPath
            break
        # Get the neighbour states of the current state
        successors = problem.getSuccessors(curState)
        for succ in successors:
            # Check whether we have been in this state before
            if succ[0] not in been:
                curPath.append(succ[1])
                # Save tuple of this neighbour state and path to reach it
                fringe.push((succ[0], list(curPath)))
                curPath.pop()
    return result

def uniformCostSearch(problem):
    """
    Searches the node of least total cost first.
    
    returns a list of actions that reaches the goal.
    """
    startState = problem.getStartState()
    # Set used to keep track of already explored nodes
    been = set()
    # Currently traversed path
    curPath = []
    fringe = util.PriorityQueue()
    # Store tuples of state and path used to reach that state
    # priority in queue is lower if less actions were needed to reach the state
    fringe.push((startState, curPath), problem.getCostOfActions(curPath))
    result = []
    while not fringe.isEmpty():
        # While the fringe is not empty, get the next tuple
        curState, curPath = fringe.pop()
        # Check whether we have been in this state before
        if curState in been: continue
        if problem.isGoalState(curState):
            # Check if state is the goal state.
            # If it is, store the current path and break the loop
            result = curPath
            break
        # Get the neighbour states of the current state
        successors = problem.getSuccessors(curState)
        for succ in successors:
            # Check whether we have been in this state before
            if succ[0] not in been:
                curPath.append(succ[1])
                # Add state in the set so that we don't reach it twice
                been.add(curState)
                # Save tuple of this neighbour state and path to reach it
                fringe.push((succ[0], list(curPath)), problem.getCostOfActions(curPath))
                curPath.pop()
    return result

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """
    Searches the node that has the lowest combined cost and heuristic first.
    
    returns a list of actions that reaches the goal.
    """
    startState = problem.getStartState()
    # Set used to keep track of already explored nodes
    been = set()
    # Currently traversed path
    curPath = []
    fringe = util.PriorityQueue()
    # Store tuples of state and path used to reach that state
    # priority in queue is lower if less actions were needed to reach the state
    # priority is also dependant on the result of heuristic function
    fringe.push((startState, curPath), problem.getCostOfActions(curPath) + heuristic(startState, problem))
    result = []
    while not fringe.isEmpty():
        # While the fringe is not empty, get the next tuple
        curState, curPath = fringe.pop()
        # Check whether we have been in this state before
        if curState in been: continue
        if problem.isGoalState(curState):
            # Check if state is the goal state.
            # If it is, store the current path and break the loop
            result = curPath
            break
        # Add state in the set so that we don't reach it twice
        been.add(curState)
        # Get the neighbour states of the current state
        successors = problem.getSuccessors(curState)
        for succ in successors:
            curPath.append(succ[1])
            # Save tuple of this neighbour state and path to reach it
            fringe.push((succ[0], list(curPath)), problem.getCostOfActions(curPath) + heuristic(succ[0], problem))
            curPath.pop()
    return result


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
