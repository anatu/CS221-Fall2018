import collections, util, copy
import itertools

############################################################
# Problem 0

# Hint: Take a look at the CSP class and the CSP examples in util.py
def create_chain_csp(n):
    # same domain for each variable
    domain = [0, 1]
    # name variables as x_1, x_2, ..., x_n
    variables = ['x%d'%i for i in range(1, n+1)]
    csp = util.CSP()
    # Problem 0c
    # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
    def XOR(a,b):
        return True if a != b else False

    for i in range(len(variables)):
        csp.add_variable(variables[i],domain)
        # Add transition factors between each element in the chain
        if i > 0:
            csp.add_binary_factor(variables[i-1],variables[i],XOR)
    # END_YOUR_CODE
    return csp



############################################################
# Problem 1

def create_nqueens_csp(n = 8):
    """
    Return an N-Queen problem on the board of size |n| * |n|.
    You should call csp.add_variable() and csp.add_binary_factor().

    @param n: number of queens, or the size of one dimension of the board.

    @return csp: A CSP problem with correctly configured factor tables
        such that it can be solved by a weighted CSP solver.
    """
    csp = util.CSP()
    # Problem 1a
    # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
    # Initialize and store the domain as a list of the rows (or columns)
    domain = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append((i,j))
        domain.append(row)

    # Utility functions for determining different row, column, or diagonal
    def isDiffRow(a, b):
        return True if a[0] != b[0] else False
    def isDiffCol(a, b):
        return True if a[1] != b[1] else False
    def isDiffDiag(a, b):
        return True if (abs(a[0] - b[0]) != abs(a[1] - b[1])) else False

    vars = []
    for i in range(n):
        # Only assign one row to each variable since the rows given to
        # previous variables automatically violate the constraints anyways
        var = "x{}".format(i)
        vars.append(var)
        csp.add_variable(var, domain[i])

    # Iterate through every possible combination of variables
    # and impose constraints for row, col, and diagonal among them
    for comb in itertools.combinations(vars, 2):
        csp.add_binary_factor(comb[0], comb[1], isDiffRow)
        csp.add_binary_factor(comb[0], comb[1], isDiffCol)
        csp.add_binary_factor(comb[0], comb[1], isDiffDiag)

    # END_YOUR_CODE
    return csp

# A backtracking algorithm that solves weighted CSP.
# Usage:
#   search = BacktrackingSearch()
#   search.solve(csp)
class BacktrackingSearch():

    def reset_results(self):
        """
        This function resets the statistics of the different aspects of the
        CSP solver. We will be using the values here for grading, so please
        do not make any modification to these variables.
        """
        # Keep track of the best assignment and weight found.
        self.optimalAssignment = {}
        self.optimalWeight = 0

        # Keep track of the number of optimal assignments and assignments. These
        # two values should be identical when the CSP is unweighted or only has binary
        # weights.
        self.numOptimalAssignments = 0
        self.numAssignments = 0

        # Keep track of the number of times backtrack() gets called.
        self.numOperations = 0

        # Keep track of the number of operations to get to the very first successful
        # assignment (doesn't have to be optimal).
        self.firstAssignmentNumOperations = 0

        # List of all solutions found.
        self.allAssignments = []

    def print_stats(self):
        """
        Prints a message summarizing the outcome of the solver.
        """
        if self.optimalAssignment:
            print "Found %d optimal assignments with weight %f in %d operations" % \
                (self.numOptimalAssignments, self.optimalWeight, self.numOperations)
            print "First assignment took %d operations" % self.firstAssignmentNumOperations
        else:
            print "No solution was found."

    def get_delta_weight(self, assignment, var, val):
        """
        Given a CSP, a partial assignment, and a proposed new value for a variable,
        return the change of weights after assigning the variable with the proposed
        value.

        @param assignment: A dictionary of current assignment. Unassigned variables
            do not have entries, while an assigned variable has the assigned value
            as value in dictionary. e.g. if the domain of the variable A is [5,6],
            and 6 was assigned to it, then assignment[A] == 6.
        @param var: name of an unassigned variable.
        @param val: the proposed value.

        @return w: Change in weights as a result of the proposed assignment. This
            will be used as a multiplier on the current weight.
        """
        assert var not in assignment
        w = 1.0
        if self.csp.unaryFactors[var]:
            w *= self.csp.unaryFactors[var][val]
            if w == 0: return w
        for var2, factor in self.csp.binaryFactors[var].iteritems():
            if var2 not in assignment: continue  # Not assigned yet
            w *= factor[val][assignment[var2]]
            if w == 0: return w
        return w

    def solve(self, csp, mcv = False, ac3 = False):
        """
        Solves the given weighted CSP using heuristics as specified in the
        parameter. Note that unlike a typical unweighted CSP where the search
        terminates when one solution is found, we want this function to find
        all possible assignments. The results are stored in the variables
        described in reset_result().

        @param csp: A weighted CSP.
        @param mcv: When enabled, Most Constrained Variable heuristics is used.
        @param ac3: When enabled, AC-3 will be used after each assignment of an
            variable is made.
        """
        # CSP to be solved.
        self.csp = csp

        # Set the search heuristics requested asked.
        self.mcv = mcv
        self.ac3 = ac3

        # Reset solutions from previous search.
        self.reset_results()

        # The dictionary of domains of every variable in the CSP.
        self.domains = {var: list(self.csp.values[var]) for var in self.csp.variables}

        # Perform backtracking search.
        self.backtrack({}, 0, 1)
        # Print summary of solutions.
        self.print_stats()

    def backtrack(self, assignment, numAssigned, weight):
        """
        Perform the back-tracking algorithms to find all possible solutions to
        the CSP.

        @param assignment: A dictionary of current assignment. Unassigned variables
            do not have entries, while an assigned variable has the assigned value
            as value in dictionary. e.g. if the domain of the variable A is [5,6],
            and 6 was assigned to it, then assignment[A] == 6.
        @param numAssigned: Number of currently assigned variables
        @param weight: The weight of the current partial assignment.
        """
        self.numOperations += 1
        assert weight > 0
        if numAssigned == self.csp.numVars:
            # A satisfiable solution have been found. Update the statistics.
            self.numAssignments += 1
            newAssignment = {}
            for var in self.csp.variables:
                newAssignment[var] = assignment[var]
            self.allAssignments.append(newAssignment)

            if len(self.optimalAssignment) == 0 or weight >= self.optimalWeight:
                if weight == self.optimalWeight:
                    self.numOptimalAssignments += 1
                else:
                    self.numOptimalAssignments = 1
                self.optimalWeight = weight

                self.optimalAssignment = newAssignment
                if self.firstAssignmentNumOperations == 0:
                    self.firstAssignmentNumOperations = self.numOperations
            return

        # Select the next variable to be assigned.
        var = self.get_unassigned_variable(assignment)
        # Get an ordering of the values.
        ordered_values = self.domains[var]

        # Continue the backtracking recursion using |var| and |ordered_values|.
        if not self.ac3:
            # When arc consistency check is not enabled.
            for val in ordered_values:
                deltaWeight = self.get_delta_weight(assignment, var, val)
                if deltaWeight > 0:
                    assignment[var] = val
                    self.backtrack(assignment, numAssigned + 1, weight * deltaWeight)
                    del assignment[var]
        else:
            # Arc consistency check is enabled.
            # Problem 1c: skeleton code for AC-3
            # You need to implement arc_consistency_check().
            for val in ordered_values:
                deltaWeight = self.get_delta_weight(assignment, var, val)
                if deltaWeight > 0:
                    assignment[var] = val
                    # create a deep copy of domains as we are going to look
                    # ahead and change domain values
                    localCopy = copy.deepcopy(self.domains)
                    # fix value for the selected variable so that hopefully we
                    # can eliminate values for other variables
                    self.domains[var] = [val]

                    # enforce arc consistency
                    self.arc_consistency_check(var)

                    self.backtrack(assignment, numAssigned + 1, weight * deltaWeight)
                    # restore the previous domains
                    self.domains = localCopy
                    del assignment[var]

    def get_unassigned_variable(self, assignment):
        """
        Given a partial assignment, return a currently unassigned variable.

        @param assignment: A dictionary of current assignment. This is the same as
            what you've seen so far.

        @return var: a currently unassigned variable.
        """

        if not self.mcv:
            # Select a variable without any heuristics.
            for var in self.csp.variables:
                if var not in assignment: return var
        else:
            # Problem 1b
            # Heuristic: most constrained variable (MCV)
            # Select a variable with the least number of remaining domain values.
            # Hint: given var, self.domains[var] gives you all the possible values.
            #       Make sure you're finding the domain of the right variable!
            # Hint: get_delta_weight gives the change in weights given a partial
            #       assignment, a variable, and a proposed value to this variable
            # Hint: for ties, choose the variable with lowest index in self.csp.variables
            # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
            # Get all the variables that haven't yet been assigned
            variables = self.csp.variables
            unassigned_vars = [var for var in self.csp.variables if var not in assignment]
            potentials = []

            # Find the sum of all the assigned values and add into list
            for var in unassigned_vars:
                sum = 0
                for val in self.domains[var]:
                    sum = sum + self.get_delta_weight(assignment, var, val)
                potentials.append((var, sum))

            # Tiebreaking
            # Compute minimum values (i.e. most constrained)
            minVal = min(potentials, key = lambda x: x[1])[1]
            # Get all variables that are most constrained 
            candidates = [elem[0] for elem in potentials if elem[1] == minVal]
            # Find their indices in the variables list
            indices = [variables.index(elem) for elem in candidates]
            # Return the tiebroken result (candidate with minimum index)
            return variables[min(indices)]
            # END_YOUR_CODE

    def arc_consistency_check(self, var):
        """
        Perform the AC-3 algorithm. The goal is to reduce the size of the
        domain values for the unassigned variables based on arc consistency.

        @param var: The variable whose value has just been set.
        """
        # Problem 1c
        # Hint: How to get variables neighboring variable |var|?
        # => for var2 in self.csp.get_neighbor_vars(var):
        #       # use var2
        #
        # Hint: How to check if a value or two values are inconsistent?
        # - For unary factors
        #   => self.csp.unaryFactors[var1][val1] == 0
        #
        # - For binary factors
        #   => self.csp.binaryFactors[var1][var2][val1][val2] == 0
        #   (self.csp.binaryFactors[var1][var2] returns a nested dict of all assignments)
        # Hint: Be careful when removing values from lists - trace through
        #       your solution to make sure it behaves as expected.


        # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
        # Create dictionary of all the unary and binary factors
        uf = self.csp.unaryFactors
        bf = self.csp.binaryFactors
        # Enforce arc consistency on a neighbor X_l w.r.t X_k
        # Return Boolean telling us whether we've modified the domain
        def enforceArcConsistency(X_1, X_2):
            # Pull domains of both variables
            domain_1 = list(self.domains[X_1])
            domain_2 = list(self.domains[X_2])
            result = False
            # Iterate through master variable
            for val_1 in domain_1:
                isConsistent = False
                # Iterate through neighbor variable
                for val_2 in domain_2:
                    # Test unary factors with respect to just X_1
                    # Check for existence in the lookup table (Ask on piazza if there's a better way to do this)
                    if uf and uf[X_1]:
                        # Check weight of factor
                        if uf[X_1][val_1] > 0:
                            isConsistent = True
                    # Test binary factors between X_1 and X_2
                    # Check existence  
                    if bf and bf[X_2] and bf[X_2][X_1] and bf[X_2][X_1][val_2]:
                        # Check weight value
                        if bf[X_2][X_1][val_2][val_1] > 0:
                            isConsistent = True
                # If no consistent assignments are found, remove
                # that value from the master variable's domain 
                if isConsistent == False:
                    result = True
                    self.domains[X_1].remove(val_1)
            return result

        queue = [var]
        queue = collections.deque(queue)
        while len(queue) > 0:
            # Remove element from queue
            X_k = queue.pop()
            # Get all neighbors
            neighbors = self.csp.get_neighbor_vars(X_k)
            # Iterate through all neighbors and check arc consistency
            for X_l in neighbors:
                # If the domain is changed for a neighbor, add it back into the queue
                if enforceArcConsistency(X_l, X_k):
                    queue.append(X_l)
        # END_YOUR_CODE


############################################################
# Problem 2b

def get_sum_variable(csp, name, variables, maxSum):
    """
    Given a list of |variables| each with non-negative integer domains,
    returns the name of a new variable with domain range(0, maxSum+1), such that
    it's consistent with the value |n| iff the assignments for |variables|
    sums to |n|.

    @param name: Prefix of all the variables that are going to be added.
        Can be any hashable objects. For every variable |var| added in this
        function, it's recommended to use a naming strategy such as
        ('sum', |name|, |var|) to avoid conflicts with other variable names.
    @param variables: A list of variables that are already in the CSP that
        have non-negative integer values as its domain.
    @param maxSum: An integer indicating the maximum sum value allowed. You
        can use it to get the auxiliary variables' domain

    @return result: The name of a newly created variable with domain range
        [0, maxSum] such that it's consistent with an assignment of |n|
        iff the assignment of |variables| sums to |n|.
    """
    # BEGIN_YOUR_CODE (our solution is 18 lines of code, but don't worry if you deviate from this)
    
    # Specify the domain of and add the result variable
    # Its domain only goes up until maxSum
    result = ("sum", name, "cumulativeSum")
    result_domain = [i for i in range(maxSum+1)]
    csp.add_variable(result, result_domain)

    # Define the helper functions used to declare the new factors
    def stepConsistency(a1, a2):
        return a1[1] == a2[0]    
    def additionConsistency(a, x):
        result = a[0] + x
        return result == a[1]
    def endConsistency(a,y):
        return a[1] == y

    # Create the domain for the A variables we are going to create
    domain = []
    for i in range(maxSum + 1):
        for j in range(maxSum + 1):
            domain.append((i,j))

    # We need to create A variables and new factors for
    # each of the original X variables in the problem
    for i in range(len(variables)):
        # Declare aux variable A_i and add to CSP
        A = ("sum", name, i)
        csp.add_variable(A, domain)
        if i == 0:
            # For the first variable, the first value should be zero
            # since we haven't added anything yet
            csp.add_unary_factor(A, lambda x: x[0] == 0)
            # Binary factor between A and X 
            csp.add_binary_factor(A, variables[i], additionConsistency)
        else:
            # Binary factor between A and X
            csp.add_binary_factor(A, variables[i], additionConsistency)
            # Binary factor between A_i and A_i-1
            prev_A = ("sum", name, i-1)
            csp.add_binary_factor(prev_A, A, stepConsistency)
    # Final binary factor to check sum value
    # Note this only checks for consistency of the sum operation,
    # it is not enforcing Sum_X <= K.
    csp.add_binary_factor(A, result, endConsistency)

    return result
    # END_YOUR_CODE

# importing get_or_variable helper function from util
get_or_variable = util.get_or_variable

############################################################
# Problem 3

# A class providing methods to generate CSP that can solve the course scheduling
# problem.
class SchedulingCSPConstructor():

    def __init__(self, bulletin, profile):
        """
        Saves the necessary data.

        @param bulletin: Stanford Bulletin that provides a list of courses
        @param profile: A student's profile and requests
        """
        self.bulletin = bulletin
        self.profile = profile

    def add_variables(self, csp):
        """
        Adding the variables into the CSP. Each variable, (request, quarter),
        can take on the value of one of the courses requested in request or None.
        For instance, for quarter='Aut2013', and a request object, request, generated
        from 'CS221 or CS246', then (request, quarter) should have the domain values
        ['CS221', 'CS246', None]. Conceptually, if var is assigned 'CS221'
        then it means we are taking 'CS221' in 'Aut2013'. If it's None, then
        we not taking either of them in 'Aut2013'.

        @param csp: The CSP where the additional constraints will be added to.
        """
        for request in self.profile.requests:
            for quarter in self.profile.quarters:
                csp.add_variable((request, quarter), request.cids + [None])

    def add_bulletin_constraints(self, csp):
        """
        Add the constraints that a course can only be taken if it's offered in
        that quarter.

        @param csp: The CSP where the additional constraints will be added to.
        """
        for request in self.profile.requests:
            for quarter in self.profile.quarters:
                csp.add_unary_factor((request, quarter), \
                    lambda cid: cid is None or \
                        self.bulletin.courses[cid].is_offered_in(quarter))

    def add_norepeating_constraints(self, csp):
        """
        No course can be repeated. Coupling with our problem's constraint that
        only one of a group of requested course can be taken, this implies that
        every request can only be satisfied in at most one quarter.

        @param csp: The CSP where the additional constraints will be added to.
        """
        for request in self.profile.requests:
            for quarter1 in self.profile.quarters:
                for quarter2 in self.profile.quarters:
                    if quarter1 == quarter2: continue
                    csp.add_binary_factor((request, quarter1), (request, quarter2), \
                        lambda cid1, cid2: cid1 is None or cid2 is None)

    def get_basic_csp(self):
        """
        Return a CSP that only enforces the basic constraints that a course can
        only be taken when it's offered and that a request can only be satisfied
        in at most one quarter.

        @return csp: A CSP where basic variables and constraints are added.
        """
        csp = util.CSP()
        self.add_variables(csp)
        self.add_bulletin_constraints(csp)
        self.add_norepeating_constraints(csp)
        return csp

    def add_quarter_constraints(self, csp):
        """
        If the profile explicitly wants a request to be satisfied in some given
        quarters, e.g. Aut2013, then add constraints to not allow that request to
        be satisfied in any other quarter. If a request doesn't specify the 
        quarter(s), do nothing.

        @param csp: The CSP where the additional constraints will be added to.
        """
        # Problem 3a
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        # # Iterate through all the requests to find quarter constraints (e.g. request XXX in Aut2016)
        for request in self.profile.requests:
            req_quarts = request.quarters
            for quarter in self.profile.quarters:
                var = (request, quarter)
                csp.add_unary_factor(var, lambda cid: cid is None or quarter in req_quarts)
        # END_YOUR_CODE

    def add_request_weights(self, csp):
        """
        Incorporate weights into the CSP. By default, a request has a weight
        value of 1 (already configured in Request). You should only use the
        weight when one of the requested course is in the solution. A
        unsatisfied request should also have a weight value of 1.

        @param csp: The CSP where the additional constraints will be added to.
        """
        for request in self.profile.requests:
            for quarter in self.profile.quarters:
                csp.add_unary_factor((request, quarter), \
                    lambda cid: request.weight if cid != None else 1.0)

    def add_prereq_constraints(self, csp):
        """
        Adding constraints to enforce prerequisite. A course can have multiple
        prerequisites. You can assume that *all courses in req.prereqs are
        being requested*. Note that if our parser inferred that one of your
        requested course has additional prerequisites that are also being
        requested, these courses will be added to req.prereqs. You will be notified
        with a message when this happens. Also note that req.prereqs apply to every
        single course in req.cids. If a course C has prerequisite A that is requested
        together with another course B (i.e. a request of 'A or B'), then taking B does
        not count as satisfying the prerequisite of C. You cannot take a course
        in a quarter unless all of its prerequisites have been taken *before* that
        quarter. You should take advantage of get_or_variable().

        @param csp: The CSP where the additional constraints will be added to.
        """
        # Iterate over all request courses
        for req in self.profile.requests:
            if len(req.prereqs) == 0: continue
            # Iterate over all possible quarters
            for quarter_i, quarter in enumerate(self.profile.quarters):
                # Iterate over all prerequisites of this request
                for pre_cid in req.prereqs:
                    # Find the request with this prerequisite
                    for pre_req in self.profile.requests:
                        if pre_cid not in pre_req.cids: continue
                        # Make sure this prerequisite is taken before the requested course(s)
                        prereq_vars = [(pre_req, q) \
                            for i, q in enumerate(self.profile.quarters) if i < quarter_i]
                        v = (req, quarter)
                        orVar = get_or_variable(csp, (v, pre_cid), prereq_vars, pre_cid)
                        # Note this constraint is enforced only when the course is taken
                        # in `quarter` (that's why we test `not val`)
                        csp.add_binary_factor(orVar, v, lambda o, val: not val or o)

    def add_unit_constraints(self, csp):
        """
        Add constraint to the CSP to ensure that the total number of units are
        within profile.minUnits/maxUnits, inclusively. The allowed range for
        each course can be obtained from bulletin.courses[cid].minUnits/maxUnits.
        For a request 'A or B', if you choose to take A, then you must use a unit
        number that's within the range of A. You should introduce any additional
        variables that you need. In order for our solution extractor to
        obtain the number of units, for every requested course, you must have
        a variable named (courseId, quarter) (e.g. ('CS221', 'Aut2013')) and
        its assigned value is the number of units.
        You should take advantage of get_sum_variable().

        @param csp: The CSP where the additional constraints will be added to.
        """
        # Problem 3b
        # Hint 1: read the documentation above carefully
        # Hint 2: the domain for each (courseId, quarter) variable should contain 0
        #         because the course might not be taken
        # Hint 3: use nested functions and lambdas like what get_or_variable and
        #         add_prereq_constraints do
        # Hint 4: don't worry about quarter constraints in each Request as they'll
        #         be enforced by the constraints added by add_quarter_constraints

        # BEGIN_YOUR_CODE (our solution is 16 lines of code, but don't worry if you deviate from this)

        bulletin = self.bulletin
        minTotalUnits = self.profile.minUnits
        maxTotalUnits = self.profile.maxUnits
        # Utility function to check if a number lies in the acceptable range
        def isInUnitRange(n):
            return (n >= minTotalUnits) and (n <= maxTotalUnits)
        # Go through each quarter in the profile
        for quarter in self.profile.quarters:
            # Go through each request in the profile
            quarter_totals = []
            for request in self.profile.requests:
                # Go through each CID in the request
                for cid in request.cids:
                    course = bulletin.courses[cid]
                    minCourseUnits = course.minUnits
                    maxCourseUnits = course.maxUnits
                    # Define the new aux variable which stores (courseId, quarter)
                    # which we will use to check total number of units as per the problem statement
                    courseVar = (cid, quarter)
                    unitRange = range(minCourseUnits, maxCourseUnits + 1)
                    # Save these variables in a list so we can use them to get the sum variable
                    quarter_totals.append(courseVar)
                    # Add the new aux variable to the CSP (the original (request, quarter) variable doesn't need to be added
                    # since it should already exist in the CSP)
                    csp.add_variable(courseVar, unitRange + [0])
                    # Define the pre-existing variable
                    requestVar = (request, quarter)
                    
                    # Utility function to make sure # units for course is correct
                    def constraintFunc(courseCode, nUnits):
                    	# Enforce units being zero if the course isn't taken that quarter
                    	if courseCode != cid:
                    		return nUnits == 0
                    	# Otherwise, make sure the number of units falls in the accepted range
                    	else:
                    		return nUnits in unitRange
                    
                    # Add constraint for units of given course
                    csp.add_binary_factor(requestVar, courseVar, constraintFunc)
            # Get the sum variable which adds all units for the quarter
            sumVar = get_sum_variable(csp = csp, name = "qtr-{}".format(quarter), variables = quarter_totals, maxSum = maxTotalUnits)
            # Impose the unary constraint on the sum variable being above the minimum and less htan the maximum allowed # of units
            csp.add_unary_factor(sumVar, isInUnitRange)


        # END_YOUR_CODE

    def add_all_additional_constraints(self, csp):
        """
        Add all additional constraints to the CSP.

        @param csp: The CSP where the additional constraints will be added to.
        """
        self.add_quarter_constraints(csp)
        self.add_request_weights(csp)
        self.add_prereq_constraints(csp)
        self.add_unit_constraints(csp)





