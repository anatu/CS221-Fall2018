import shell
import util
import wordsegUtil

############################################################
# Problem 1b: Solve the segmentation problem under a unigram model

# Define the problem which will be passed as "problem"
# argument into UniformCostSearch via the segmentWords function below
# (see util.py for UCS implementation) 
# Query is a string
# unigramCost is the cost function
class SegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    # Define the start state (e.g. for tram, it's start at position 1)
    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
        return self.query
        # END_YOUR_CODE

    # Return a boolean expression stating whether the end state has been reached
    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        return len(state) == 0
        # END_YOUR_CODE

    # Return a set of tuples which describe successive actions that can be taken along with the cost 
    # of each action
    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        result = []
        segments = [state[0:j] for j in range(1,len(state)+1)]
        for segment in segments:
        	# New state is the old state minus the segment (first occurrence only, since in 
        	# the state space model we start at the left side of the state string)
            result.append((segment, state.replace(segment, "", 1), self.unigramCost(segment)))
        return result
        # END_YOUR_CODE

def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=1)
    ucs.solve(SegmentationProblem(query, unigramCost))
    
    # BEGIN_YOUR_CODE (our solution is 10 lines of code, but don't worry if you deviate from this)
    result = " ".join(ucs.actions)
    return result
    # END_YOUR_CODE



############################################################
# Problem 2b: Solve the vowel insertion problem under a bigram cost
class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return (0, "-BEGIN-")
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        return (state[0] == len(self.queryWords)) & (state != (0, "-BEGIN-"))
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 16 lines of code, but don't worry if you deviate from this)
        result = []
        fillset = self.possibleFills(self.queryWords[state[0]])
        for fill in fillset:
            newState = (state[0] + 1, fill)
            result.append((fill, newState, self.bigramCost(state[1], fill)))
        return result
        # END_YOUR_CODE
        
# queryWords = input sequence of words, bigramCost = cost function, possibleFills = function that returns
# list of possible reconstructions for a given word
def insertVowels(queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    if len(queryWords) == 0:
    	return ""

    ucs = util.UniformCostSearch(verbose = 1)
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))
    if ucs.actions != None:
        return " ".join(ucs.actions)
    else:
        return " ".join(queryWords)
    # END_YOUR_CODE


############################################################
# Problem 3b: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query, bigramCost, possibleFills):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        return (0, wordsegUtil.SENTENCE_BEGIN)
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        return (state[0] == len(self.query)) & (state != (0, wordsegUtil.SENTENCE_BEGIN))
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 23 lines of code, but don't worry if you deviate from this)
        # Two moving windows to split up the word, populate vowels, and compute cost
        result = []
        
        substr = self.query[state[0]:]
        for i in range(len(substr)+1):
            fill_set = self.possibleFills(substr[:i])
            
            if fill_set == set([]): continue
            
            for fill in fill_set:
                newState = (state[0] + i, fill)
                result.append((fill, newState, self.bigramCost(state[1], fill)))
                
        return result
                    
        # END_YOUR_CODE

def segmentAndInsert(query, bigramCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_CODE (our solution is 11 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch(verbose = 1)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))

    if ucs.actions != None:
        return " ".join(ucs.actions)
    else:
        return query
    # END_YOUR_CODE

############################################################

if __name__ == '__main__':
    shell.main()
