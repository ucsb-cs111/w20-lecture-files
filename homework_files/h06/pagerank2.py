def pagerank2(E, return_vector = False, max_iters = 1000, tolerance = 1e-6):
    """compute page rank from sparse adjacency matrix

    Inputs:
      E: adjacency matrix with links going from cols to rows.
         E is a matrix of 0s and 1s, where E[i,j] = 1 means 
         that web page (vertex) j has a link to web page i.
      return_vector = False: If True, return the eigenvector as well as the ranking.
      max_iters = 1000: Maximum number of power iterations to do.
      tolerance = 1e-6: Stop when the eigenvector norm changes by less than this.
      
    Outputs:
      ranking: Permutation giving the ranking, most important first.
      vector (only if return_vector is True): Dominant eigenvector of PageRank matrix.

    This computes page rank by the following steps:
    1. Add links from any dangling vertices to all vertices.
    2. Scale the columns to sum to 1.
    3. Add a constant matrix to represent jumping at random 15% of the time.
    4. Find the dominant eigenvector with the power method.
    5. Sort the eigenvector to get the rankings.
    
    This computes the same page rank as pagerank1, but it never creates
    a new matrix as large as E.
    Instead, it computes the matrix-vector product M @ v in the power method
    in several steps corresponding to the steps in converting E to M.
    """

    # sparse and dense matrices do some things differently, sigh...
    if type(E) is not scipy.sparse.csr.csr_matrix:
        print('Warning, converting input from type', type(E), 'to sparse csr_matrix.')
        E = sparse.csr_matrix(E)

    nnz = E.count_nonzero()
    outdegree = np.array(E.sum(axis = 0))[0]
    nrows, n = E.shape

    assert nrows == n, 'E must be square'
    assert np.max(E) == 1 and np.sum(E) == nnz, 'E must contain only zeros and ones'
    
    #  1. Add links from any dangling vertices to all other vertices.
    #     We don't add any actual links or compute the matrix F with full columns, 
    #     but just compute the vector "d" that picks out the nonzero cols of F.
    #     Then, formally, F = (J - I) @ D,
    #     where J is the all-ones matrix, I is the identity, and D is diag(d),
    #     but we don't ever form I or J or D or F explicitly.
    #
    #  2. Scale the columns of E + F to sum to 1.
    #     Again we don't compute a whole matrix, just a vector "t"
    #     for which the columns of (E + F) @ diag(t) sum to 1
    #     so, formally, A = (E + F) @ T,
    #     where F is as above and T = diag(t).
    #     Each element of t is one over a column sum of E or of F.
    #
    #  We do both steps (1) and (2) in the same loop below.
   
    t = np.zeros(n)
    d = np.zeros(n) 
    for j in range(n):
        if outdegree[j] == 0:
            t[j] = 1 / (n-1)
            d[j] = 1
        else:
            t[j] = 1 / outdegree[j]
            d[j] = 0
    
    
    #  3. Add a constant matrix to represent jumping at random 15% of the time.
    #     Again we don't do this explicitly, just get the ingredients for:
    #         M = (1-m) * A + m * J / n

    m = 0.15
    
    #  4. Find the dominant eigenvector.
    #     Here we use the power method, just as in pagerank1, but we
    #     implement the matrix-vector multiplication M @ v in several steps.
    
    #  Start with v as a vector all of whose entries are 1/n.

    e = np.ones(n)
    v = e / npla.norm(e)

    for iteration in range(max_iters):
        oldv = v
             
        # Now  M @ v = (1-m)*(E + F) @ T @ v  +  m/n * J @ v.
        #
        # If we let w = T @ v = v * t, and note that J @ v = np.sum(v)*e, we get
        #   M @ v = (1-m) * E @ w + (1-m) * F @ w + m/n * np.sum(v)*e, 
        # and since F @ w = J @ D @ w - I @ D @ w = np.sum(w*d) * e - w*d, we finally have
        #   M @ v = (1-m)*(E@w) + (1-m)*np.sum(w*d)*e - (1-m)*w*d + (m/n)*np.sum(v)*e,
        # which requires no matrices except E, multiplying the vector w.
        
        w  = v * t                   # * is elementwise multiply; v, t, w are vectors 
        wd = w * d                   # elementwise multiply again; w, d, and wd are vectors
        v1 = (1-m) * (E @ w)         # the only matrix is the original E, times vector w
        v2 = (1-m) * np.sum(wd) * e  # scalar times vector e
        v3 = (1-m) * wd              # scalar times vector w
        v4 = (m/n) * np.sum(v) * e   # scalar times vector e
        v = v1 + v2 - v3 + v4        # adding and subtracting vectors                 
                    
        eigval = npla.norm(v)
        v = v / eigval
        
        if npla.norm(v - oldv) < tolerance:
            break
    
    if npla.norm(v - oldv) < tolerance:
        print('Dominant eigenvalue is %f after %d iterations.\n' % (eigval, iteration+1))
    else:
        print('Did not converge to tolerance %e after %d iterations.\n' % (tolerance, max_iters))

    # Check that the eigenvector elements are all the same sign, and make them positive
    assert np.all(v > 0) or np.all(v < 0), 'Error: eigenvector is not all > 0 or < 0'
    vector = np.abs(v)
        
    #  5. Sort the eigenvector and reverse the permutation to get the rankings.
    ranking = np.argsort(vector)[::-1]

    if return_vector:
        return ranking, vector
    else:
        return ranking
