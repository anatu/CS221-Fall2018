import collections, sys, os
from logic import *

############################################################
# Problem 1: propositional logic
# Convert each of the following natural language sentences into a propositional
# logic formula.  See rainWet() in examples.py for a relevant example.

# Sentence: "If it's summer and we're in California, then it doesn't rain."
def formula1a():
    # Predicates to use:
    Summer = Atom('Summer')               # whether it's summer
    California = Atom('California')       # whether we're in California
    Rain = Atom('Rain')                   # whether it's raining
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return Implies(And(Summer, California), Not(Rain))
    # END_YOUR_CODE

# Sentence: "It's wet if and only if it is raining or the sprinklers are on."
def formula1b():
    # Predicates to use:
    Rain = Atom('Rain')              # whether it is raining
    Wet = Atom('Wet')                # whether it it wet
    Sprinklers = Atom('Sprinklers')  # whether the sprinklers are on
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return Equiv(Or(Rain, Sprinklers), Wet)    
    # END_YOUR_CODE

# Sentence: "Either it's day or night (but not both)."
def formula1c():
    # Predicates to use:
    Day = Atom('Day')     # whether it's day
    Night = Atom('Night') # whether it's night
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return Xor(Day, Night)
    # END_YOUR_CODE

############################################################
# Problem 2: first-order logic

# Sentence: "Every person has a mother."
def formula2a():
    # Predicates to use:
    def Person(x): return Atom('Person', x)        # whether x is a person
    def Mother(x, y): return Atom('Mother', x, y)  # whether x's mother is y

    # Note: You do NOT have to enforce that the mother is a "person"
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    personName = "$personName"
    motherExists = Exists('$candidateMother', Mother(personName, '$candidateMother'))
    return Forall(personName, Implies(Person(personName), motherExists))
    # END_YOUR_CODE

# Sentence: "At least one person has no children."
def formula2b():
    # Predicates to use:
    def Person(x): return Atom('Person', x)        # whether x is a person
    def Child(x, y): return Atom('Child', x, y)    # whether x has a child y

    # Note: You do NOT have to enforce that the child is a "person"
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    personName = "$personName"
    childName = "$childName"
    return Exists(personName, And(Person(personName), Not(Exists(childName, Child(personName,childName)))))
    # END_YOUR_CODE

# Return a formula which defines Daughter in terms of Female and Child.
# See parentChild() in examples.py for a relevant example.
def formula2c():
    # Predicates to use:
    def Female(x): return Atom('Female', x)            # whether x is female
    def Child(x, y): return Atom('Child', x, y)        # whether x has a child y
    def Daughter(x, y): return Atom('Daughter', x, y)  # whether x has a daughter y
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    personName = "$personName"
    childName = "$childName"    
    # Form 1 - using Female structure
    femaleChildCheck =  And(Child(personName, childName), Female(childName))
    # Form 2 - using Daughter structure
    childIsDaughter = Daughter(personName, childName)
    # Equivalence check across all parent-child pairs pairs
    return Forall(personName, Forall(childName, Equiv(Daughter(personName, childName), femaleChildCheck)))
    # END_YOUR_CODE

# Return a formula which defines Grandmother in terms of Female and Parent.
# Note: It is ok for a person to be her own parent
def formula2d():
    # Predicates to use:
    def Female(x): return Atom('Female', x)                  # whether x is female
    def Parent(x, y): return Atom('Parent', x, y)            # whether x has a parent y
    def Grandmother(x, y): return Atom('Grandmother', x, y)  # whether x has a grandmother y
    # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
    parentName = "$parentName"
    childName = "$childName"    
    gmName = "$grandmotherName"
    # Form 1 - using Grandmother
    grandmotherCheck = Grandmother(parentName, childName)
    # Form 2 - using two Parents
    parentOfParentCheck = Exists(gmName, AndList([Female(childName), Parent(parentName, gmName), Parent(gmName, childName)]))
    return Forall(parentName, Forall(childName, Equiv(grandmotherCheck, parentOfParentCheck)))
    # END_YOUR_CODE

############################################################
# Problem 3: Liar puzzle

# Facts:
# 0. John: "It wasn't me!"
# 1. Susan: "It was Nicole!"
# 2. Mark: "No, it was Susan!"
# 3. Nicole: "Susan's a liar."
# 4. Exactly one person is telling the truth.
# 5. Exactly one person crashed the server.
# Query: Who did it?
# This function returns a list of 6 formulas corresponding to each of the
# above facts.
# Hint: You might want to use the Equals predicate, defined in logic.py.  This
# predicate is used to assert that two objects are the same.
# In particular, Equals(x,x) = True and Equals(x,y) = False iff x is not equal to y.
def liar():
    def TellTruth(x): return Atom('TellTruth', x)
    def CrashedServer(x): return Atom('CrashedServer', x)
    john = Constant('john')
    susan = Constant('susan')
    nicole = Constant('nicole')
    mark = Constant('mark')
    formulas = []
    # We provide the formula for fact 0 here.
    formulas.append(Equiv(TellTruth(john), Not(CrashedServer(john))))
    # You should add 5 formulas, one for each of facts 1-5.
    # BEGIN_YOUR_CODE (our solution is 11 lines of code, but don't worry if you deviate from this)
    pA = "$pA"
    pB = "$pB"
    # We know that if any true, then all others false (since only one can tell truth)
    # John says "it wasn't me" -> If true, then he didn't crash the server (defined above)
    # Susan says "It was Nicole!" -> If true, then Nicole crashed server     
    susanTruth = Equiv(TellTruth(susan), CrashedServer(nicole))
    formulas.append(susanTruth)
    # Mark says "it was susan" -> If true, then it was susan 
    markTruth = Equiv(TellTruth(mark), CrashedServer(susan))
    formulas.append(markTruth)
    # Nicole says "Susan's a liar" -> If true, then Susan is not telling the truth
    nicoleTruth = Equiv(TellTruth(nicole), Not(TellTruth(susan)))
    formulas.append(nicoleTruth)
    # One person is telling the truth
    def truthEquivalence(pA, pB):
        # There is one person pA for which the equivalence of truth-telling is False
        # for all other pB's (i.e. pA telling the truth, all pB's are not)
        equiv = Equiv(Not(Equals(pA, pB)),Not(TellTruth(pB)))
        return Forall(pB, equiv)

    formulas.append(Exists(pA, And(TellTruth(pA), truthEquivalence(pA,pB))))
    # One person crashed the server
    def serverCrashEquivalence(pA, pB):
        # There is one person pA for which equivalence of CrasherServer is false
        # for all other pB's (pA crashed server, nobody else did)
        equiv =  Equiv(Not(Equals(pA, pB)),Not(CrashedServer(pB)))
        return Forall(pB, equiv)
    formulas.append(Exists(pA, And(CrashedServer(pA), serverCrashEquivalence(pA,pB))))    
    # END_YOUR_CODE
    query = CrashedServer('$x')
    return (formulas, query)

############################################################
# Problem 5: Odd and even integers

# Return the following 6 laws:
# 0. Each number $x$ has a unique successor, which is not equal to $x$.
# 1. Each number is either even or odd, but not both.
# 2. The successor number of an even number is odd.
# 3. The successor number of an odd number is even.
# 4. For every number $x$, the successor of $x$ is larger than $x$.
# 5. Larger is a transitive property: if $x$ is larger than $y$ and $y$ is
#    larger than $z$, then $x$ is larger than $z$.
# Query: For each number, there exists an even number larger than it.
def ints():
    def Even(x): return Atom('Even', x)                  # whether x is even
    def Odd(x): return Atom('Odd', x)                    # whether x is odd
    def Successor(x, y): return Atom('Successor', x, y)  # whether x's successor is y
    def Larger(x, y): return Atom('Larger', x, y)        # whether x is larger than y
    # Note: all objects are numbers, so we don't need to define Number as an
    # explicit predicate.
    # Note: pay attention to the order of arguments of Successor and Larger.
    # Populate |formulas| with the 6 laws above and set |query| to be the
    # query.
    # Hint: You might want to use the Equals predicate, defined in logic.py.  This
    # predicate is used to assert that two objects are the same.
    formulas = []
    query = None
    # BEGIN_YOUR_CODE (our solution is 30 lines of code, but don't worry if you deviate from this)
    # Define variable names
    x = "$x"
    y = "$y"
    z = "$z"
    # 1. Each number x has exactly one successor, which is not equal to x.
    succNotEqual_1 = And(Successor(x,y),Not(Equals(x,y)))
    formula_1 = Equiv(succNotEqual_1, Equals(z,y))
    list_1 = [Successor(x,z),Not(Equals(x,z)),Forall(y, formula_1)]
    result_1 = Forall(x, Exists(z,AndList(list_1)))
    formulas.append(result_1)
    # 2. Each number is either odd or even, but not both.
    # Either odd and not even, or even and not odd, for all variables
    result_2 = Forall(x,Or(And(Even(x),Not(Odd(x))),And(Odd(x),Not(Even(x)))))
    formulas.append(result_2)
    # 3. The successor of an even number is odd.    
    # Even variable with successor
    evenSuccessor_3 = And(Even(x),Successor(x,y))
    # Implication that the successor is odd
    result_3 = Forall(x, Forall(y, Implies(evenSuccessor_3,Odd(y))))
    formulas.append(result_3)
    # 4. The successor of an odd number is even.
    # Odd variable with successor
    oddSuccessor_4 = And(Odd(x),Successor(x,y))
    # Implication that the successor is even
    result_4 = Forall(x, Forall(y, Implies(oddSuccessor_4,Even(y))))
    formulas.append(result_4)
    # 5. For every number x, the successor of x is larger than x.
    # Generic successor
    successor_5 = Successor(x,y)
    # Successor is larger
    result_5 = Forall(x, Forall(y,Implies(successor_5,Larger(y, x))))
    formulas.append(result_5)
    # 6. Larger is a transitive property: if x is larger than y and y is larger than z, then x is larger than z.
    xBiggerThanY_6 = Larger(x,y)
    yBiggerThanZ_6 = Larger(y,z)
    xBiggerThanYZ_6 = And(xBiggerThanY_6, yBiggerThanZ_6) 
    result_6 = Forall(x, Forall(y, Forall(z, (Implies(xBiggerThanYZ_6,Larger(x,z))))))
    formulas.append(result_6)
    # END_YOUR_CODE
    query = Forall("$x", Exists('$y', And(Even('$y'), Larger('$y', '$x'))))
    return (formulas, query)

############################################################
# Problem 6: semantic parsing (extra credit)
# Each of the following functions should return a GrammarRule.
# Look at createBaseEnglishGrammar() in nlparser.py to see what these rules should look like.
# For example, the rule for 'X is a Y' is:
#     GrammarRule('$Clause', ['$Name', 'is', 'a', '$Noun'],
#                 lambda args: Atom(args[1].title(), args[0].lower()))
# Note: args[0] corresponds to $Name and args[1] corresponds to $Noun.
# Note: by convention, .title() should be applied to all predicates (e.g., Cat).
# Note: by convention, .lower() should be applied to constant symbols (e.g., garfield).

from nlparser import GrammarRule

def createRule1():
    # Return a GrammarRule for 'every $Noun $Verb some $Noun'
    # Note: universal quantification should be outside existential quantification.
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    x = "$x"
    clause = "$Clause"
    split = "every $Noun $Verb some $Noun".split(" ")
    def func(args):
        return Forall(x, Implies(Atom(args[0].title(), x), Atom(args[1].title(), x)))
    result = GrammarRule(clause, split, func)
    return result
    # END_YOUR_CODE

def createRule2():
    # Return a GrammarRule for 'there is some $Noun that every $Noun $Verb'
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    x = "$x"
    clause = "$Clause"
    split = "there is some $Noun that every $Noun $Verb".split(" ")
    def func(args):
        return Forall(x, Implies(Atom(args[0].title(), x), Atom(args[1].title(), x)))
    result = GrammarRule(clause, split, func)
    return result
    # END_YOUR_CODE

def createRule3():
    # Return a GrammarRule for 'if a $Noun $Verb a $Noun then the former $Verb the latter'
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    x = "$x"
    clause = "$Clause"
    split = 'if a $Noun $Verb a $Noun then the former $Verb the latter'.split(" ")
    def func(args):
        return Forall(x, Implies(Atom(args[0].title(), x), Atom(args[1].title(), x)))
    result = GrammarRule(clause, split, func)
    return result
    # END_YOUR_CODE