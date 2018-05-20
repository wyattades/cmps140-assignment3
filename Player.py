import numpy as np
from numpy import inf
import time


class Board:

    def __init__(self, player_number, board_matrix=None):
        self.player = 0
        self.opponent = 0
        self.player_number = player_number
        self.opponent_number = 1 if player_number == 2 else 2

        if board_matrix is not None:
            player, opponent = '', ''
            for col in range(6, -1, -1):
                player += '0'
                opponent += '0'
                for row in range(0, 6):
                    player += '1' if board_matrix[row, col] == self.player_number else '0'
                    opponent += '1' if board_matrix[row, col] == self.opponent_number else '0'
            self.player = int(player, 2)
            self.opponent = int(opponent, 2)

    def valid_moves(self):
        mask = self.player | self.opponent
        moves = []
        for col in range(7):
            if (mask & (1 << ((col * 7) + 5))) == 0:
                moves.append(col)
        return moves

    def move(self, col, player_number):
        new = Board(self.player_number)
        mask = self.player | self.opponent
        mask |= mask + (1 << (col * 7))
        if player_number == self.player_number:
            new.player = self.opponent ^ mask 
            new.opponent = self.opponent
        else:
            new.player = self.player
            new.opponent = self.player ^ mask
        return new

    def get(self, row, col):
        select = 1 << ((col * 7) + 5 - row)
        if self.player & select != 0: return self.player_number
        if self.opponent & select != 0: return self.opponent_number
        return 0

    def is_end(self):

        # Board is full
        if (self.player | self.opponent) == 279258638311359: # this number represents the value of a full bitboard
            return 0
        
        # Check 4 in a row
        for p, n in [ (self.player, self.player_number), (self.opponent, self.opponent_number) ]:
            # This strategy for checking 4 in a row is borrowed from:
            # https://medium.com/@gillesvandewiele/creating-the-perfect-connect-four-ai-bot-c165115557b0
            
            # Horizontal -
            m = p & (p >> 7)
            if m & (m >> 14):
                return n
            # Diagonal \
            m = p & (p >> 6)
            if m & (m >> 12):
                return n
            # Diagonal /
            m = p & (p >> 8)
            if m & (m >> 16):
                return n
            # Vertical |
            m = p & (p >> 1)
            if m & (m >> 2):
                return n

        return 0


class AIPlayer:

    def __init__(self, player_number):
        self.player_number = player_number
        self.opponent_number = 1 if player_number == 2 else 2
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)

    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                    the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        MAX_DEPTH = 6

        # Create bitstring representation of board to improve performance
        board = Board(self.player_number, board)

        def max_value(board, alpha, beta, depth):
            if depth >= MAX_DEPTH or board.is_end():
                return self.evaluation_function(board)
            v = -inf
            for col in board.valid_moves():
                v = max(v, min_value(board.move(col, self.player_number), alpha, beta, depth + 1))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(board, alpha, beta, depth):
            if depth >= MAX_DEPTH or board.is_end():
                return self.evaluation_function(board)
            v = inf
            for col in board.valid_moves():
                v = min(v, max_value(board.move(col, self.opponent_number), alpha, beta, depth + 1))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        start_time = time.time()

        best_score = -inf
        best_col = None
        for col in board.valid_moves():
            v = min_value(board.move(col, self.player_number), best_score, inf, 0)
            if v > best_score:
                best_score = v
                best_col = col

        print('Alpha-Beta Depth={} finished in {} seconds'.format(MAX_DEPTH, time.time() - start_time))
        
        return best_col

    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                    the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        # Must generate random seed because multiprocessing always uses same seed???
        np.random.seed()

        MAX_DEPTH = 4

        # Create bitstring representation of board to improve performance
        board = Board(self.player_number, board)

        def value(board, depth, agent):
            if depth >= MAX_DEPTH or board.is_end(): return self.evaluation_function(board)
            if agent: return max_value(board, depth + 1)
            else: return exp_value(board, depth + 1)

        def max_value(board, depth):
            return max([ value(board.move(col, self.player_number), depth, False) for col in board.valid_moves() ])

        def exp_value(board, depth):
            return np.mean([ value(board.move(col, self.opponent_number), depth, True) for col in board.valid_moves() ])

        start_time = time.time()

        best_score = -inf
        best_col = None
        for col in board.valid_moves():
            score = value(board.move(col, self.player_number), 0, False)
            if score > best_score:
                best_score = score
                best_col = col

        print('Expectimax Depth={} finished in {} seconds'.format(MAX_DEPTH, time.time() - start_time))

        return best_col

    ROWS, COLS = 6, 7

    # (delta_x, delta_y, range_x, range_y)
    DELTAS = [
        (1, 0, range(ROWS - 3), range(ROWS)), # Horizontal (-) score
        (0, 1, range(COLS), range(ROWS - 3)), # Vertical (|) score
        (1, 1, range(COLS - 3), range(ROWS - 3)), # Diagonal (\) score
        (-1, 1, range(3, COLS), range(ROWS - 3)), # Diagonal (/) score
    ]


    def evaluation_function(self, board):
        """
        Given the current stat of the board, return the scalar value that 
        represents the evaluation function for the current player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                    the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """

        four = range(4)
        total_score = 0

        # Iterate over ALL possible 4-in-a-row sequences on the game-board
        for delta_x, delta_y, range_x, range_y in self.DELTAS:
            for col in range_x:
                for row in range_y:
                    x = col
                    y = row
                    this_score = 0
                    other_score = 0
                    for _ in four:
                        val = board.get(y, x)
                        # Check if there's a piece below this one
                        # below = y + 1 >= self.ROWS or board.get(y + 1, x) != 0
                        # if below:
                        if val == self.player_number: this_score += 1
                        elif val == self.opponent_number: other_score += 1
                        x += delta_x
                        y += delta_y

                    # Only give points for this possibility if the other player has
                    # no pieces obstructing it
                    if other_score == 0:
                        # Add this player's score
                        if this_score == 2: total_score += 1
                        elif this_score == 3: total_score += 3
                        elif this_score == 4: total_score += 7
                    elif this_score == 0:
                        # Subtract opponent's score
                        if other_score == 2: total_score -= 1
                        elif other_score == 3: total_score -= 3
                        elif other_score == 4: total_score -= 7

        return total_score


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                    the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        np.random.seed()
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:,col]:
                valid_cols.append(col)

        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                    the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move

