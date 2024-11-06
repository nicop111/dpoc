"""ProblemSet3_Sol8.py
Python script that implements the Smith-Waterman algorithm for sequence alignment
(https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm)
which is a special case of the DPA. In bioinformatics, a sequence alignment
is a way of arranging the sequences of DNA to identify regions of similarity
(https://en.wikipedia.org/wiki/Sequence_alignment).
--
Dynamic Programming and Optimal Control
Problem Set 3, Problem 8
"""
import numpy as np


def mismatch_penalty(a, b):
    """Return the mismatch penalty between a and b.

    Each base substitution or amino acid substitution is assigned a
    score. In general, matches are assigned negative costs, and
    mismatches are assigned relatively higher costs.

    Parameters
    ----------
    a : character
        sequence element 1
    b : chaaracter
        sequence element 2

    Returns
    -------
    int
        similarity score
    """
    # TODO
    return -3 if a == b else 3
    # END TODO


def gap_penalty(k):
    """Return the gap penalty for a gap of length k.

    The gap penalty assigns penalties for insertion or deletion.
    We use a linear penalty for other possibilities see
    https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm.

    Parameters
    ----------
    k : int
        gap length

    Returns
    -------
    int
        gap penalty
    """
    # TODO
    return 2 * k
    # END TODO


def cost_of_alignment(alignment):
    """Return the cost of an alignment.

    Parameters
    ----------
    alignment : List of tuples of characters
        e.g. [('A','B'), ...]

    Returns
    -------
    int
        Cost of the alignment
    """
    cost = 0
    i = 0
    # Do not score leading gaps
    if i < len(alignment):
        while alignment[i][0] == "-" or alignment[i][1] == "-":
            i += 1
            if i >= len(alignment):
                break
    while i < len(alignment):
        gap_length = 0
        while alignment[i][0] == "-" or alignment[i][1] == "-":
            gap_length += 1
            i += 1
            if i >= len(alignment):
                break
        # Do not score trailing gaps
        if i >= len(alignment):
            break
        if gap_length == 0:
            cost += mismatch_penalty(ord(alignment[i][0]), ord(alignment[i][1]))
            i += 1
        else:
            cost += gap_penalty(gap_length)
    return cost


def smith_waterman_algorithm(s1, s2, verbose=True):
    """Apply the Smith-Waterman algorithm to the sequences s1, s2.

    Parameters
    ----------
    s1 : string
        genome sequence 1
    s2 : string
        genome sequence 2
    verbose : bool, optional
        Print cost and input matrix, by default True

    Returns
    -------
    int
        Score of the alignment
    """
    # Add 1 to the number of rows and columns for the ghost cells.
    # These are used to allow starting the alignment at any position.
    rows = len(s1) + 1
    cols = len(s2) + 1

    # The cost matrix contains the cost to go for each state.
    cost_matrix = np.zeros((rows, cols))
    # The input matrix contains the optimal input for each state.
    input_matrix = np.zeros((rows, cols))

    # TODO: Fill the cost matrix and save the optimal input.
    for row in range(1, rows):
        for col in range(1, cols):
            # Compute the costs for gaps in sequence 1.
            left_costs = np.array(
                [cost_matrix[row][col - i] + gap_penalty(i) for i in range(1, col + 1)]
            )
            # Compute the costs for gaps in sequence 2.
            up_costs = np.array(
                [cost_matrix[row - i][col] + gap_penalty(i) for i in range(1, row + 1)]
            )

            costs = np.array(
                [
                    0,
                    cost_matrix[row - 1, col - 1]
                    + mismatch_penalty(s1[row - 1], s2[col - 1]),
                    left_costs.min(),
                    up_costs.min(),
                ]
            )
            cost_matrix[row, col] = costs.min()

            if costs.argmin() == 2:
                # Gap lengths in sequence 1 are stored as positive integers
                input_matrix[row, col] = left_costs.argmin() + 1
            elif costs.argmin() == 3:
                # Gap lengths in sequence 2 are stored as negative integers
                input_matrix[row, col] = -(up_costs.argmin() + 1)
            # else we just leave the input at zero
    # END TODO
    if verbose:
        print("Cost matrix:")
        print(cost_matrix)
        print("Input matrix:")
        print(input_matrix)

    # The alignment should contain a list of tupels. Use '-' for gaps.
    # e.g. [('A','A'),('G','C'),('C','-'),('T','-')]
    # There can be multiple optimal alignments. We only return one alignment
    # and the associated optimal cost.
    alignment = []
    # TODO: Perform traceback to find the optimal alignment based on the cost and
    # input matrix.
    # Start the traceback at the state with minimum value.
    row, col = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
    # Proceed to the next state according to the optimal input until a zero value
    # is encountered.
    while cost_matrix[row, col] != 0:
        input = int(input_matrix[row, col])
        if input < 0:
            for i in range(-input):
                alignment.append((s1[row - 1], "-"))
                row -= 1
        elif input > 0:
            for i in range(input):
                alignment.append(("-", s2[col - 1]))
                col -= 1
        else:
            alignment.append((s1[row - 1], s2[col - 1]))
            row -= 1
            col -= 1
    # Reverse the alignment because the traceback proceeds backwards.
    alignment.reverse()
    # END TODO
    cost = cost_of_alignment(alignment)
    print(
        "Alignment with cost "
        + str(cost)
        + " for the strings "
        + s1
        + " and "
        + s2
        + ":"
    )
    header = "+".join(["-" * 5 for _ in alignment])
    print("+" + header + "+")
    first_row = "|".join(["{:^5}".format(tup[0]) for tup in alignment])
    print("|" + first_row + "|")
    print("+" + header + "+")
    second_row = "|".join(["{:^5}".format(tup[1]) for tup in alignment])
    print("|" + second_row + "|")
    print("+" + header + "+")
    return cost


if __name__ == "__main__":
    # An optimal alignment for the following strings  has a score of -13
    cost1 = smith_waterman_algorithm("TGTTACGG", "GGTTGACTA")
    # An optimal alignment for the following strings  has a score of -15
    cost2 = smith_waterman_algorithm("TACGGGCCCGCTAC", "TAGCCCTATCGGTCA", verbose=False)

    print("TESTS PASSED: " + str(cost1 == -13 and cost2 == -15))

    in_strings = (
        "Type in two strings separated by a white space or press enter to terminate\n"
    )
    in_strings = input(in_strings)
    while in_strings:
        print(
            "Cost",
            smith_waterman_algorithm(
                *in_strings.replace('"', "").replace("'", "").split()
            ),
        )
        in_strings = "Type in two strings separated by a white space or press enter to terminate\n"
        in_strings = input(in_strings)
