import numpy as np

class AIPlayer:

    MAX_DEPTH = 4

    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)

    def _valid_moves(self, board):
        [ rows, cols ] = board.shape
        moves = []
        for col in range(cols):
            if board[0][col] == 0:
                for row in range(rows):
                    if row + 1 >= rows or board[row + 1][col] != 0:
                        moves.append((row, col))
                        break
        return moves

    def _get_alpha_beta_extreme(self, board, depth, alpha, beta, extreme):
        score = self.evaluation_function(board)
        
        if depth <= 0 or abs(score) == np.Infinity: return (None, score)

        [ rows, cols ] = board.shape
        extreme_col = None
        extreme_score = -extreme * np.Infinity

        for row, col in self._valid_moves(board):

            board[row][col] = self.player_number
            (_, next_score) = self._get_alpha_beta_extreme(board, depth - 1, alpha, beta, -extreme)
            board[row][col] = 0

            if extreme == 1:
                if next_score > extreme_score:
                    extreme_score = next_score
                    extreme_col = col
                if extreme_score >= beta: break
                if extreme_score > alpha: alpha = extreme_score
            else:
                if next_score < extreme_score:
                    extreme_score = next_score
                    extreme_col = col
                if extreme_score <= alpha: break
                if extreme_score < beta: beta = extreme_score
                
                # (next_col, next_score) = self._get_alpha_beta_extreme(board, depth + 1, alpha, beta, -extreme)
                # if extreme_col == None or next_score > extreme_score if extreme == 1 else next_score < extreme_score:
                #     extreme_col = col
                #     extreme_score = alpha = next_score
                # if alpha >= beta: break
        
        return (extreme_col, extreme_score)

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

        (col, _) = self._get_alpha_beta_extreme(board, self.MAX_DEPTH, -np.Infinity, np.Infinity, 1)
        print('')
        return col

    def _board_end(self, board):
        player_win_str = '{0}{0}{0}{0}'.format(self.player_number)
        to_str = lambda a: ''.join(a.astype(str))

        def check_horizontal(b):
            for row in b:
                if player_win_str in to_str(row):
                    return True
            return False

        def check_verticle(b):
            return check_horizontal(b.T)

        def check_diagonal(b):
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b
                
                root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                if player_win_str in to_str(root_diag):
                    return True

                for i in range(1, b.shape[1]-3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int))
                        if player_win_str in diag:
                            return True

            return False

        return (check_horizontal(board) or
                check_verticle(board) or
                check_diagonal(board))

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

        def value(board, depth, agent):
            if depth <= 0 or self._board_end(board): return self.evaluation_function(board)
            if agent: return max_value(board, depth - 1)[1]
            else: return exp_value(board, depth - 1)[1]

        def max_value(board, depth):
            max_score = -np.Infinity
            max_col = None

            for row, col in self._valid_moves(board):
                board[row][col] = self.player_number
                score = value(board, depth, False)
                board[row][col] = 0

                if score > max_score:
                    max_score = score
                    max_col = col

            return max_col, max_score

        def exp_value(board, depth):
            choices = []

            for row, col in self._valid_moves(board):
                board[row][col] = self.player_number
                choices.append((col, value(board, depth, True)))
                board[row][col] = 0

            return choices[np.random.randint(len(choices))]

        return max_value(board, self.MAX_DEPTH)[0]


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

        [ rows, cols ] = board.shape
        
        total_score = 0
        four = range(4)

        # (delta_x, delta_y, range_x, range_y)
        deltas = [
            (1, 0, range(cols - 3), range(rows)), # Horizontal (-) score
            (0, 1, range(cols), range(rows - 3)), # Vertical (|) score
            (1, 1, range(cols - 3), range(rows - 3)), # Diagonal (\) score
            (-1, 1, range(3, cols), range(rows - 3)), # Diagonal (/) score
        ]

        print(board)

        for delta_x, delta_y, range_x, range_y in deltas:
            for col in range_x:
                for row in range_y:
                    x = col
                    y = row
                    this_score = 0
                    other_score = 0
                    for _ in four:
                        val = board[y][x]
                        if val == self.player_number:
                            this_score += 1
                        elif val != 0:
                            other_score += 1
                            if other_score >= 4: print(other_score)
                        x += delta_x
                        y += delta_y
                    if this_score == 4:
                        total_score += 1
                    #     # return 10000
                    #     return np.Infinity
                    if other_score == 4:
                        print('dsfsdf')
                        return 0
                    #     # return 10000
                    #     return -np.Infinity
                    # if this_score > 0 and other_score == 0:
                    #     total_score += this_score
                    # if other_score > 0 and this_score == 0:
                    #     total_score -= other_score

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

