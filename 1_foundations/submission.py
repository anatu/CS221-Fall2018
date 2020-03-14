import collections
import math

############################################################
# Problem 3a

def findAlphabeticallyLastWord(text):
    """
    Given a string |text|, return the word in |text| that comes last
    alphabetically (that is, the word that would appear last in a dictionary).
    A word is defined by a maximal sequence of characters without whitespaces.
    You might find max() and list comprehensions handy here.
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return max(text.split(" "))
    # END_YOUR_CODE

############################################################
# Problem 3b

def euclideanDistance(loc1, loc2):
    """
    Return the Euclidean distance between two locations, where the locations
    are pairs of numbers (e.g., (3, 5)).
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return math.sqrt(sum([(a - b)**2 for a,b in zip(loc1, loc2)]))
    # END_YOUR_CODE

############################################################
# Problem 3c

def mutateSentences(sentence):
    """
    Given a sentence (sequence of words), return a list of all "similar"
    sentences.
    We define a sentence to be similar to the original sentence if
      - it as the same number of words, and
      - each pair of adjacent words in the new sentence also occurs in the original sentence
        (the words within each pair should appear in the same order in the output sentence
         as they did in the orignal sentence.)
    Notes:
      - The order of the sentences you output doesn't matter.
      - You must not output duplicates.
      - Your generated sentence can use a word in the original sentence more than
        once.
    Example:
      - Input: 'the cat and the mouse'
      - Output: ['and the cat and the', 'the cat and the mouse', 'the cat and the cat', 'cat and the cat and']
                (reordered versions of this list are allowed)
    """
    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    def compute_mutations(starter, candidates):
        if (len(starter) == len(candidates) + 1):
            return starter
        for candidate in candidates:
            if starter[-1] == candidate[0]:
                result = starter + [candidate[1]]
                return compute_mutations(result, candidates)        
               
    bigrams = []
    split_sent = sentence.lower().split(" ")
    for i in range(0, len(split_sent)-1):
        bigrams.append([split_sent[i], split_sent[i + 1]])

    mutated_sents = []
    mutated_sents.append(sentence.lower())
    for starter in bigrams:
        final = compute_mutations(starter, bigrams)
        if final is not None:
            mutated_sents.append(" ".join(final))

    return list(set(mutated_sents)) 		    
    # END_YOUR_CODE

############################################################
# Problem 3d

def sparseVectorDotProduct(v1, v2):
    """
    Given two sparse vectors |v1| and |v2|, each represented as collections.defaultdict(float), return
    their dot product.
    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    dot_prod = 0
    for key in set.intersection(set(v1.keys()), set(v2.keys())):
        dot_prod = dot_prod + (v1[key] * v2[key])
    return dot_prod
    # END_YOUR_CODE

############################################################
# Problem 3e

def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    for key in v2.keys():
        v1[key] = v1[key] + scale*v2[key]
    return v1
    # END_YOUR_CODE

############################################################
# Problem 3f

def findSingletonWords(text):
    """
    Splits the string |text| by whitespace and returns the set of words that
    occur exactly once.
    You might find it useful to use collections.defaultdict(int).
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    splittext = text.split(" ")
    return set([i for i in splittext if splittext.count(i) == 1])
    # END_YOUR_CODE

############################################################
# Problem 3g

def computeLongestPalindromeLength(text):
    n = len(text) 
    
    # Catch empty strings
    if n == 0:
        return 0

    table = [[0 for x in range(n)] for x in range(n)] 
  
    # strings with length 1 are palindromic by default 
    for i in range(n): 
        table[i][i] = 1
    
    # Start from the outside and work inwards on the string
    # This way our window will both universally move inwards on the whole string,
    # as well as increment through all of the possible substrings within our window
    for substring in range(2, n+1): 
        for i in range(n-substring+1): 
            end_index = i + substring - 1
            # Start / end are the same and length == 2. Therefore palindrome
            if (text[i] == text[end_index]) and (substring == 2): 
                table[i][end_index] = 2
            # Start / end are the same but length is greater than 2.
            # Simply add 2 to whatever the solution of one step "inside" is
            # so we can "discard" the same outside letters but keep the memory
            # of their contribution to the palindrome length
            elif text[i] == text[end_index]: 
                table[i][end_index] = table[i+1][end_index-1] + 2
            # If they are not equal, we increment both sides of the window to see
            # which has a greater length and pass that as the result
            else: 
                table[i][end_index] = max(table[i][end_index-1], table[i+1][end_index]); 
    
    # Due to the incremental summation, we take the end value of the first row
    # as the length of the longest substring
    return table[0][n-1] 
    # END_YOUR_CODE
