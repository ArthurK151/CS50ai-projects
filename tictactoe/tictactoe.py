"""
Tic Tac Toe Player
"""

import copy
import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # X always moves first, so equal counts means X's turn; more X's means O's turn
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)
    return O if x_count > o_count else X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()

    # Collect (row, col) coordinates for every unoccupied cell
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                possible_actions.add((i, j))

    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action

    # Reject out-of-bounds indices or moves onto already-occupied cells
    if i not in range(3) or j not in range(3) or board[i][j] != EMPTY:
        raise ValueError("Invalid action for the given board.")

    # Deep copy prevents Minimax search branches from mutating the caller's board
    new_board = copy.deepcopy(board)
    new_board[i][j] = player(board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Check each row and column for three matching non-empty marks
    for idx in range(3):
        if board[idx][0] == board[idx][1] == board[idx][2] != EMPTY:
            return board[idx][0]
        if board[0][idx] == board[1][idx] == board[2][idx] != EMPTY:
            return board[0][idx]

    # Check both main diagonals for three-in-a-row
    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != EMPTY:
        return board[0][2]

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # Game ends immediately on a win or once every cell is filled
    if winner(board) is not None:
        return True
    return all(cell != EMPTY for row in board for cell in row)


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    game_winner = winner(board)
    if game_winner == X:
        return 1
    if game_winner == O:
        return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # Terminal boards have no valid next move
    if terminal(board):
        return None

    current_player = player(board)

    # X maximizes utility while O minimizes utility
    if current_player == X:
        _, best_action = _max_value(board)
    else:
        _, best_action = _min_value(board)

    return best_action


def _max_value(board):
    """
    Helper returning (best_score, best_action) for maximizing player X.
    """
    if terminal(board):
        return utility(board), None

    best_score = -math.inf
    best_action = None

    for action in actions(board):
        score, _ = _min_value(result(board, action))
        if score > best_score:
            best_score = score
            best_action = action
            # Stop searching further siblings if X already found a guaranteed win (+1)
            if best_score == 1:
                break

    return best_score, best_action


def _min_value(board):
    """
    Helper returning (best_score, best_action) for minimizing player O.
    """
    if terminal(board):
        return utility(board), None

    best_score = math.inf
    best_action = None

    for action in actions(board):
        score, _ = _max_value(result(board, action))
        if score < best_score:
            best_score = score
            best_action = action
            # Stop searching further siblings if O already found a guaranteed win (-1)
            if best_score == -1:
                break

    return best_score, best_action