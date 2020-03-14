from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions():
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """

    # BEGIN_YOUR_CODE (our solution is 26 lines of code, but don't worry if you deviate from this)
    
    legalMoves = gameState.getLegalActions(self.index)

    # Check game end
    def isEnd(gameState, legalActions):
        if len(legalActions) == 0:
          return True
        elif gameState.isWin():
          return True 
        elif gameState.isLose():
          return True
        else:
          return False

    # Return the optimal move by searching through the
    # tree up to the depth limit and applying min/max behavior
    # for the relevant agents
    def compute_action(depth, agentIndex, gameState):
        # Start over if the max number of agents is reached
        if agentIndex == gameState.getNumAgents():
            agentIndex = 0
        # Decrement depth layer by one after starting over
        # (not before otherwise will influence the min / max behavior
        # below for the last ghost)
        if agentIndex == 0:
            depth = depth - 1


        legalActions = gameState.getLegalActions(agentIndex)

        # Stop when terminal state is reached
        if isEnd(gameState,legalActions):
            return (gameState.getScore(), Directions.STOP)

        # Stop search at maximum depth, and just return evaluation function result
        # at max depth the action will always be to stop
        if depth == 0:
            return (self.evaluationFunction(gameState), Directions.STOP)

        # Evaluate the score of all possible successor states using
        # the case structure defined above
        result = []
        for action in legalActions:
            succState = gameState.generateSuccessor(agentIndex,action)
            move = compute_action(depth,agentIndex+1,succState)
            result.append((move[0],action))

        # Minimize if agent is a ghost (based on utility)
        # Each ghost will minimize separately on the set of successor actions
        if agentIndex > 0:
            best_action = min(result, key = lambda t: t[0])
        # Maximize if agent is pacman (based on utility)
        if agentIndex == 0:
            best_action = max(result, key = lambda t: t[0])
        # Return both utility and the action for pass-through use in recursion
        # (the final return doesn't actually need the utility)
        return best_action

    (utility, action) = compute_action(self.depth + 1, self.index, gameState)
    return action
    # END_YOUR_CODE

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    # BEGIN_YOUR_CODE (our solution is 49 lines of code, but don't worry if you deviate from this)
    legalMoves = gameState.getLegalActions(self.index)

    # Check game end
    def isEnd(gameState, legalActions):
        if len(legalActions) == 0:
          return True
        elif gameState.isWin():
          return True 
        elif gameState.isLose():
          return True
        else:
          return False

    # Return the optimal move by searching through the
    # tree up to the depth limit and applying min/max behavior
    # for the relevant agents
    def compute_action(depth, agentIndex, gameState):
        # Start over if the max number of agents is reached
        if agentIndex == gameState.getNumAgents():
            agentIndex = 0
        # Decrement depth layer by one after starting over
        # (not before otherwise will influence the min / max behavior
        # below for the last ghost)
        if agentIndex == 0:
            depth = depth - 1


        legalActions = gameState.getLegalActions(agentIndex)

        # Stop when terminal state is reached
        if isEnd(gameState,legalActions):
            return (gameState.getScore(), Directions.STOP)

        # Stop search at maximum depth, and just return evaluation function result
        # at max depth the action will always be to stop
        if depth == 0:
            return (self.evaluationFunction(gameState), Directions.STOP)

        # Evaluate the score of all possible successor states using
        # the case structure defined above
        result = []
        inf = float("inf")
        alpha = -inf
        beta = inf 
        for action in legalActions:
            succState = gameState.generateSuccessor(agentIndex,action)
            move = compute_action(depth,agentIndex+1,succState)
            # Update alpha if agent is the maximizing player (agent)\
            if agentIndex == 0:
              value = move[0]
              alpha = max(value, alpha)
            # Else update beta for the minimizing agents (ghosts)
            else:
              value = move[0]
              beta = min(value, beta)
            # Stop looking through successor states if the pruning condition is met
            # (i.e. prune if beta < alpha, indicating no overlap with the maximum-outcome ancestor)
            if beta < alpha:
              break
            result.append((move[0],action))

        # Minimize if agent is a ghost (based on utility)
        # Each ghost will minimize separately on the set of successor actions
        if agentIndex > 0:
            best_action = min(result, key = lambda t: t[0])
        # Maximize if agent is pacman (based on utility)
        if agentIndex == 0:
            best_action = max(result, key = lambda t: t[0])
        # Return both utility and the action for pass-through use in recursion
        # (the final return doesn't actually need the utility)
        return best_action

    (utility, action) = compute_action(self.depth + 1, self.index, gameState)
    print utility
    return action
    # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_CODE (our solution is 25 lines of code, but don't worry if you deviate from this)
    # Everything about the function should be the same except for the optimizing behavior
    # of the agents themselves
    legalMoves = gameState.getLegalActions(self.index)

    # Check game end
    def isEnd(gameState, legalActions):
        if len(legalActions) == 0:
          return True
        elif gameState.isWin():
          return True 
        elif gameState.isLose():
          return True
        else:
          return False

    # Return the optimal move by searching through the
    # tree up to the depth limit and applying min/max behavior
    # for the relevant agents
    def compute_action(depth, agentIndex, gameState):
        # Start over if the max number of agents is reached
        if agentIndex == gameState.getNumAgents():
            agentIndex = 0
        # Decrement depth layer by one after starting over
        # (not before otherwise will influence the min / max behavior
        # below for the last ghost)
        if agentIndex == 0:
            depth = depth - 1


        legalActions = gameState.getLegalActions(agentIndex)

        # Stop when terminal state is reached
        if isEnd(gameState,legalActions):
            return (gameState.getScore(), Directions.STOP)

        # Stop search at maximum depth, and just return evaluation function result
        # at max depth the action will always be to stop
        if depth == 0:
            return (self.evaluationFunction(gameState), Directions.STOP)

        # Evaluate the score of all possible successor states using
        # the case structure defined above
        result = []
        for action in legalActions:
            succState = gameState.generateSuccessor(agentIndex,action)
            move = compute_action(depth,agentIndex+1,succState)
            result.append((move[0],action))

        # Random selection if agent is a ghost 
        if agentIndex > 0:
            # Choose a random action
            best_action = random.choice(result)
            # Compute the expected utility (random distribution)
            E_util = sum([elem[0] for elem in result])/len(result)
            # Overwrite the utility of the random action with the expected utility
            best_action = (E_util, best_action[1])
        # Maximize if agent is pacman 
        if agentIndex == 0:
            best_action = max(result, key = lambda t: t[0])
        # Return both utility and the action for pass-through use in recursion
        # (the final return doesn't actually need the utility)
        return best_action

    (utility, action) = compute_action(self.depth + 1, self.index, gameState)
    return action
    # END_YOUR_CODE

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
    Your extreme, unstoppable evaluation function (problem 4).

    DESCRIPTION: <write something here so we know what you did>
  """

  # BEGIN_YOUR_CODE (our solution is 26 lines of code, but don't worry if you deviate from this)
  
  ######################################
  ######################################
  # PREPROCESSING
  ######################################
  ######################################
  final_score = 0
  # Base score of the current state (i.e. the "naive" evaluation function)
  base_score = currentGameState.getScore()
  final_score = base_score 

  # Get the positions of all the relevant game objects in the given states
  pacman_pos = currentGameState.getPacmanPosition()
  ghosts_pos = currentGameState.getGhostStates()
  pellets = currentGameState.getFood().asList()
  capsules = currentGameState.getCapsules()


  ######################################
  ######################################
  # UTILITY FUNCTIONS
  ######################################
  ######################################
  # Output the distance to the closest capsule from pacman
  def findClosestPellet(pacman_pos, pellets):
    dists = [] 
    for pellet in pellets:
      dists.append(util.manhattanDistance(pacman_pos, pellet))
    # May be zero pellets on board
    if len(dists) > 0:
      return min(dists)
    else:
      return 1

  # Find out if we're really close to a capsule! 
  def isNearCapsule(pacman_pos, capsules):
    dists = []
    for capsule in capsules:
      dists.append(util.manhattanDistance(pacman_pos, capsule))
    if len(capsules) > 0:
      return min(dists)
    else:
      return 1000

  # Closest ghost from pacman
  def findClosestGhost(pacman_pos, ghosts_pos):
    dists = []
    for ghost in ghosts_pos:
      dists.append(util.manhattanDistance(pacman_pos, ghost.getPosition()))
    # There should never be zero ghosts 
    return min(dists)

  # Return distnace to food centroid
  def findFoodCentroid(pacman_pos, pellets):
    if len(pellets) == 0:
      return None
    cent_x = 0
    cent_y = 0
    for pellet in pellets:
      cent_x += pellet[0]
      cent_y += pellet[1]
    cent_x = cent_x / len(pellets)
    cent_y = cent_y / len(pellets)
    return util.manhattanDistance(pacman_pos, (cent_x, cent_y))

  def foodDists(pacman_pos, pellets):
    dists = []
    for pellet in pellets:
      dists.append(util.manhattanDistance(pacman_pos, pellet))
    return sum(dists)

  ######################################
  ######################################
  # CASE STRUCTURE FOR BUILDING SCORE
  ######################################
  ######################################
  safetyMargin = 3
  
  final_score = final_score - 1/findClosestGhost(pacman_pos, ghosts_pos)
  final_score = final_score + findClosestPellet(pacman_pos, pellets)

  # Augment score if pacman is closer to food than a ghost
  if (findClosestPellet(pacman_pos, pellets)) < (findClosestGhost(pacman_pos, ghosts_pos) + safetyMargin):
    final_score = final_score*2

  # Augment score if we're near a capsule'
  if isNearCapsule(pacman_pos, capsules) < (findClosestGhost(pacman_pos, ghosts_pos) + safetyMargin):
    final_score = final_score*1.1


  final_score = -0.2*foodDists(pacman_pos, pellets)

  return final_score

  # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction