# tied_rank and auc are from https://github.com/benhamner/Metrics
# with LICENSE:
# Copyright (c) 2012, Ben Hamner
# Author: Ben Hamner (ben@benhamner.com)
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: 
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer. 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution. 
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

def tied_rank(x):
    """
    Computes the tied rank of elements in x.

    This function computes the tied rank of elements in x.

    Parameters
    ----------
    x : list of numbers, numpy array

    Returns
    -------
    score : list of numbers
            The tied rank f each element in x

    """
    sorted_x = sorted(zip(x,range(len(x))))
    r = [0 for k in x]
    cur_val = sorted_x[0][0]
    last_rank = 0
    for i in range(len(sorted_x)):
        if cur_val != sorted_x[i][0]:
            cur_val = sorted_x[i][0]
            for j in range(last_rank, i): 
                r[sorted_x[j][1]] = float(last_rank+1+i)/2.0
            last_rank = i
        if i==len(sorted_x)-1:
            for j in range(last_rank, i+1): 
                r[sorted_x[j][1]] = float(last_rank+i+2)/2.0
    return r

def tied_rank_ooc(sorted_x):
    """
    Computes the tied rank of elements in SORTED x.

    This function computes the tied rank of elements in x.

    Parameters
    ----------
    sorted_x : generator of numbers NEVER EVER GIVING None

    Returns
    -------
    score : generator of
            The tied rank f each element in x

    """
    last_val = None
    next_rank = 1
    tied = False

    from itertools import chain
    for val in chain(sorted_x, (None for x in xrange(1))):
        if last_val is not None:
            if tied:
                if last_val == val:
                    tied_count += 1
                else:
                    rank = float(sum(xrange(next_rank, next_rank + tied_count))) / tied_count
                    for x in xrange(tied_count):
                        yield rank
                    tied = False
                    next_rank += tied_count
                    tied_count = 0
            elif last_val == val:
                tied = True
                tied_count = 2
            else:
                # just output as normal
                yield float(next_rank)
                next_rank += 1
        else:
            # it's the beginning, do nothing
            pass

        last_val = val

def auc(actual, posterior):
    """
    Computes the area under the receiver-operater characteristic (AUC)

    This function computes the AUC error metric for binary classification.

    Parameters
    ----------
    actual : list of binary numbers, numpy array
             The ground truth value
    posterior : same type as actual
                Defines a ranking on the binary numbers, from most likely to
                be positive to least likely to be positive.

    Returns
    -------
    score : double
            The mean squared error between actual and posterior

    """
    r = tied_rank(posterior)
    num_positive = len([0 for x in actual if x==1])
    num_negative = len(actual)-num_positive
    sum_positive = sum([r[i] for i in range(len(r)) if actual[i]==1])
    auc = ((sum_positive - num_positive*(num_positive+1)/2.0) /
           (num_negative*num_positive))
    return auc

def auc_ooc(actual, posterior, posterior_length):
    """
    Computes the area under the receiver-operater characteristic (AUC)

    This function computes the AUC error metric for binary classification.

    actual and posterior must be sorted.

    Parameters
    ----------
    actual : list of binary numbers, numpy array
             The ground truth value
    posterior : same type as actual
                Defines a ranking on the binary numbers, from most likely to
                be positive to least likely to be positive.

    Returns
    -------
    score : double
            The mean squared error between actual and posterior

    """
    r = tied_rank_ooc(posterior)
    num_positive = 0
    num_negative = 0
    sum_positive = 0.0

    actual_and_rank = ((x, r.next()) for x in actual)   
    for x, rank in actual_and_rank:
        if x == 1:
            num_positive += 1
            sum_positive += rank
        else:
            num_negative += 1

    #num_positive = sum([1 for x in actual if x==1])
    #num_negative = len(actual)-num_positive
    #sum_positive = sum([r[i] for i in range(len(r)) if actual[i]==1])
    auc = ((sum_positive - num_positive*(num_positive+1)/2.0) /
           (num_negative*num_positive))
    return auc

if __name__ == '__main__':
    print tied_rank([1,2,2,3,3,5])
    print [x for x in tied_rank_ooc([1,2,2,3,3,5])]
    #print auc([0,0,1,1,0,1],[1,2,2,3,5,6])
    #print auc_ooc([0,0,1,1,0,1],[1,2,2,3,5,6], 6)

    """
[1.0, 2.5, 2.5, 4.0]
[1.0, 2.5, 2.5, 4.0]
0.722222222222
0.722222222222
    """ # from :r !python %

    """
"""

""" broken:
posterior:
0.488540
0.588689
0.311658
0.379008
0.224716
0.522872
0.530129
0.446732
0.700344
0.304563
actual:
1
-1
-1
-1
1
-1
-1
-1
1
-1
AUC 0.52380952381
