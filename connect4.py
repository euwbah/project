from calendar import c
import os
import random
import math
from shutil import copyfile
import shutil
from time import sleep, time
from typing import List, Optional, Tuple

'''
NOTE 1: The `board` list that represents board state will have 7 * c elements,
        where `c` is the number of columns.

        The first 7 elements in the list represent the **bottom** row of the board,
        the next 7 represent the row above it, and so on.

NOTE 2: Since all functions are immutable/have no side effects, this should be a
        purely functional solution, i.e. no global mutable variables are required.
'''

# Enumerating player color and number.
# Noughts goes first.
NOUGHTS = 1 # (Yellow)
CROSSES = 2 # (Red)

COLS = 7 # There's always 7 columns according to implementation notes.

MINMAX_DEPTH = 4 # Recommended by assignment spec.


def next_player(who_played: int) -> int:
    '''
    Returns the next player/opponent given the current player.

    ### Parameters

    - `who_played`: The current player

    ### Returns

    The next player ID.
    '''

    # NOTE: no support for >= 3 players yet

    assert 1 <= who_played <= 2, 'Unsupported player'

    return NOUGHTS if who_played == CROSSES else CROSSES


def check_move(board: List[int], turn: int, col: int, pop: bool) -> bool:
    '''
    Checks if a certain move is valid given a `board` state. Returns whether or not move is valid.

    A valid move entails:
    - Dropping a piece in a column that is not already full
    - Popping a piece from a column of which the bottom piece belongs to the current player
      AND isn't empty.

    ### Parameters
        - `board`: the board state
        - `turn`: which player makes this move
        - `col`: the column to drop/pop the piece. This is zero-indexed (first col is 0).
        - `pop`: `True` if pop, `False` if drop
    
    ### Returns
        `True` if move is valid, `False` otherwise
    '''

    # Checking whether column is valid is to be done in the menu game UI code, not here.
    # If user-input is out-of-bounds here, this is an unexpected error.
    assert 0 <= col < COLS, 'Invalid column'
    
    col_pieces = board[col::7] # represents pieces of given `col` from bottom to top.
    
    # A move is valid if it is Drop (not pop) and the column is not full (topmost piece in column 0),
    return (not pop and col_pieces[-1] == 0) \
        or (pop and col_pieces[0] == turn) # or if it is Pop and the bottom piece belongs to the current player.


def check_stalemate(board: List[int], turn: int) -> bool:
    '''
    Checks if the given player is in a stalemate (has no legal moves left).

    This will only return `True` in the very rare case that the board is full AND
    all the pieces at the bottom are the opponent's pieces, not allowing the current
    player to pop or drop anything. However it's still a probable case, so it should
    be accounted for.

    ### Parameters

    - `board`: The current state of the board
    - `turn`: Which player to check for stalemate

    ### Returns

    `True` if current `turn` player has no legal moves left. `False` otherwise.
    '''
    for col in range(COLS):
        for pop in [True, False]:
            if check_move(board, turn, col, pop):
                return False
    
    return True
    

def apply_move(board: List[int], turn: int, col: int, pop: bool) -> List[int]:
    '''
    Applies a given move to the board. Returns the new board state.

    This is an **immutable** function with NO SIDE EFFECTS, i.e. the list
    referred to by the `board` variable is not modified. if-else loop. 1) drop 2) pop

    ### Parameters
        - `board`: the board state
        - `turn`: which player makes this move
        - `col`: the column to drop/pop the piece. This is zero-indexed (first col is 0).
        - `pop`: `True` if pop, `False` if drop

    ### Returns
        The new board state (list of ints)
    '''
    
    # represents the updated board to return
    board_copy = board.copy()
    
    if pop == False:
        # zero-indexed row number of the first occurence of 0 in given `col`
        row = next((i for i, x in enumerate(board[col::7]) if x == 0), None)
        assert row is not None, 'Invalid move! Column is full.' # should never happen
        board_copy[row * COLS + col] = turn
    else: # pop == True
        col_pieces = board[col::7] # represents pieces of given `col` from bottom to top.
        # pop the bottom piece, shift all contents down one row
        # replace topmost piece with 0.
        board_copy[col::7] = col_pieces[1:] + [0]
        
    return board_copy


def check_victory(board: List[int], who_played: int) -> int:
    '''
    Checks if a player has won the game. Returns the player who won, or 0 if no one has won.

    ||----------------------------------------------------------------------------------||
    ||NOTE: According to telegram chat, if some player somehow makes a move such that   ||
    ||the board would be in a winning position for BOTH players (e.g. via a pop move),  ||
    ||the player that made the move LOSES.                                              ||
    ||----------------------------------------------------------------------------------||

    ### Parameters
        - `board`: the board state
        - `who_played`: the player who just made a move

    ### Returns
        The player number of the player who won, or 0 if no one has won.
        If the board is in a position that is winning for both players, then return
        the OPPONENT player. The player who just made such a move loses.

        I.e. you lose if you make a move that would win the game for your opponent, even
        if it is also winning for yourself.
    '''

    assert 1 <= who_played <= 2, "Unsupported player number"

    # NOTE: these two have to be separate variables as we have to consider
    #       the case where both players win and we have to confer the win to the
    #       opponent of `who_played`.
    #
    #       this also means we cannot do early termination unless we reach a case where
    #       both players wins, then the result is fully certain.
    noughts_wins = False
    crosses_wins = False
    num_rows = len(board) // COLS

    # Check horizontals (left to right)
    for row in range(num_rows):
        # Check if there are 4 consecutive pieces of the same color
        # in a row.

        streak_piece = 0 # the current player number which has N pieces in a row. 0 means no player.
        index_of_streak_start = 0 # the beginning index of the N pieces in a row

        for col in range(COLS):
            piece = board[row * 7 + col]
            if piece != streak_piece: # streak is broken
                if streak_piece != 0 and col - index_of_streak_start == 4:
                    if streak_piece == NOUGHTS:
                        noughts_wins = True
                    elif streak_piece == CROSSES: # redundant elif, but here for future-proofing multiplayers if needed.
                        crosses_wins = True
                index_of_streak_start = col # reset streak index
                streak_piece = piece
        
        # Check if row ended with a winning streak:
        if streak_piece != 0 and COLS - index_of_streak_start == 4:
            if streak_piece == NOUGHTS:
                noughts_wins = True
            elif streak_piece == CROSSES:
                crosses_wins = True
    
    # Checking verticals and diagonals only make sense if num_rows >= 4
    if num_rows >= 4:
        # Check verticals (bottom to top)
        for col in range(COLS):
            # Check if there are 4 consecutive pieces of the same color
            # in a column.

            streak_piece = 0
            index_of_streak_start = 0

            for row in range(num_rows):
                piece = board[row * 7 + col]
                if piece != streak_piece:
                    if streak_piece != 0 and row - index_of_streak_start == 4:
                        if streak_piece == NOUGHTS:
                            noughts_wins = True
                        elif streak_piece == CROSSES:
                            crosses_wins = True
                    index_of_streak_start = row
                    streak_piece = piece

            # Check if end of column has a winning streak:
            if streak_piece != 0 and num_rows - index_of_streak_start == 4:
                if streak_piece == NOUGHTS:
                    noughts_wins = True
                elif streak_piece == CROSSES:
                    crosses_wins = True

        # Check up-left diagonals (bottom-right to top-left)
        
        # contains all starting bottom-right points such that diagonals have at least
        # 4 pieces in them.
        starting_coords = [(0, x) for x in range(3, COLS)]
        starting_coords += [(x, COLS - 1) for x in range(1, num_rows - 3)]

        # traverse one diagonal at a time from the above starting points
        for row, col in starting_coords:
            streak_piece = 0
            index_of_streak_start = 0
            diagonal_idx = 0 # The (n+1)th piece of the current diagonal

            while row + diagonal_idx < num_rows and col - diagonal_idx >= 0:
                piece = board[(row + diagonal_idx) * 7 + col - diagonal_idx]
                if piece != streak_piece:
                    if streak_piece != 0 and diagonal_idx - index_of_streak_start == 4:
                        if streak_piece == NOUGHTS:
                            noughts_wins = True
                        elif streak_piece == CROSSES:
                            crosses_wins = True
                    index_of_streak_start = diagonal_idx
                    streak_piece = piece
                diagonal_idx += 1

            # Check if the last few pieces are a winning streak:
            if streak_piece != 0 and diagonal_idx - index_of_streak_start == 4:
                if streak_piece == NOUGHTS:
                    noughts_wins = True
                elif streak_piece == CROSSES:
                    crosses_wins = True

        # Check up-right diagonals (bottom-left to top-right)

        # similar to above, contains all starting bottom-left points such that diagonals have at least
        # 4 pieces in them.

        starting_coords = [(0, x) for x in range(COLS - 4, -1, -1)]
        starting_coords += [(x, 0) for x in range(1, num_rows - 3)]

        for row, col in starting_coords:
            streak_piece = 0
            index_of_streak_start = 0
            diagonal_idx = 0

            while row + diagonal_idx < num_rows and col + diagonal_idx < COLS:
                piece = board[(row + diagonal_idx) * 7 + col + diagonal_idx]
                if piece != streak_piece:
                    if streak_piece != 0 and diagonal_idx - index_of_streak_start == 4:
                        if streak_piece == NOUGHTS:
                            noughts_wins = True
                        elif streak_piece == CROSSES:
                            crosses_wins = True
                    index_of_streak_start = diagonal_idx
                    streak_piece = piece
                diagonal_idx += 1

            # Check if the last few pieces are a winning streak:
            if streak_piece != 0 and diagonal_idx - index_of_streak_start == 4:
                if streak_piece == NOUGHTS:
                    noughts_wins = True
                elif streak_piece == CROSSES:
                    crosses_wins = True

    if noughts_wins and crosses_wins:
        return next_player(who_played)
    elif noughts_wins:
        return NOUGHTS
    elif crosses_wins:
        return CROSSES
    
    return 0 # nobody wins


def find_immediate_win(board: List[int], turn: int) -> Optional[Tuple[int, bool]]:
    '''
    Finds a legal move that can be made by current player's `turn` that will
    immediately win the game for the current player.

    ### Parameters

    - `board`: The current state of the board
    - `turn`: Which player to find winning moves for.

    ### Returns

    Either `(column: int, pop: bool)` if such a winning move can be found or
    `None` if no such move can be found.
    '''
    for col in range(COLS):
        for pop in [True, False]:
            if not check_move(board, turn, col, pop):
                continue

            board_copy = apply_move(board, turn, col, pop)

            if check_victory(board_copy, turn) == turn:
                # Found a winning move, return it.
                return col, pop
    
    # No winning moves
    return None


def find_immediate_win_multiple(board: List[int], turn: int) -> List[Tuple[int, bool]]:
    '''
    Finds a list of all the legal moves that can be made by current player's `turn` that will
    immediately win the game for the current player.

    ### Parameters

    - `board`: The current state of the board
    - `turn`: Which player to find winning moves for.

    ### Returns

    A list of `[(column: int, pop: bool)]` tuples, each one representing a possible winning move.
    There may be 0, 1, or many winning moves, depending on the board.
    '''

    winning_moves = []

    for col in range(COLS):
        for pop in [True, False]:
            if not check_move(board, turn, col, pop):
                continue

            board_copy = apply_move(board, turn, col, pop)

            if check_victory(board_copy, turn) == turn:
                # Found a winning move, append it to the list
                winning_moves.append((col, pop))
    
    # No winning moves
    return winning_moves


def eval_cps(board: List[int]) -> float:
    '''
    Evaluates the CPS score metric (see README.md)
    
    This metric is stateless and doesn't depend on whose turn it is
    currently. It measures the winning opportunities of each player 
    (Noughts winning will contribute a positive score, crosses winning
    will contribute a negative score)

    ### Parameters

    - `board`: The current state of the board
    '''

    '''
    Naive brute force method:

    Give Nought 3 free moves.
    
    For every move where Nought can win within the 1st free move, add 1000 points.
    If within the 2nd free move, add 100 points.
    If within the 3rd free move, add 10 points.

    Similarly, give Cross 3 free moves (reset the board state to the input argument).

    Subtract 1000, 100, and 10 points for every move where Cross can win within the 1st, 2nd, and 3rd 
    free move respectively (just as above).

    The final score tally represents who is winning.

    Note: this is computationally heavy as there are 14^3 = 2744 permutations of 3 free moves.
    '''

    tally = 0

    def recurse_free_moves(board: List[int], depth_remaining: int, turn: int) -> float:
        '''
        Recursively iterates through all permutations of free moves and returns CPS score obtained by
        current iteration.

        ### Parameters

        - `board`: The current state of the board
        - `depth_remaining`: How many free moves are left to be made (1 represents last free move, terminate recursion)
        - `turn`: Which player is to make the free moves

        ### Returns

        Positive/absolute CPS score obtained by current iteration.
        '''

        assert depth_remaining > 0
        
        # Find all immediately winning moves for current player

        winning_moves = find_immediate_win_multiple(board, turn)
        cps = 10 ** depth_remaining * len(winning_moves)

        # Iterate legal moves and recurse each scenario

        if depth_remaining == 1:
            # No deeper recursion, return CPS score
            return cps

        for col in range(COLS):
            for pop in [True, False]:
                if not check_move(board, turn, col, pop):
                    continue

                board_copy = apply_move(board, turn, col, pop)
                cps += recurse_free_moves(board_copy, depth_remaining - 1, turn)

        return cps

    tally += recurse_free_moves(board, 3, NOUGHTS)
    tally -= recurse_free_moves(board, 3, CROSSES)

    return tally


def decide_minimax_move(board: List[int], turn: int, depth: int) -> Tuple[int, Optional[bool]]:
    '''
    Finds the best move for the current player using the minimax algorithm.

    ### Parameters

    - `board`: The current state of the board
    - `turn`: Which player to find the best move for.
    - `depth`: How many half-turns ahead to look.

    ### Returns

    The best move for the current player, in the form `(column: int, pop: bool)`.
    Returns `None` if stalemate and no legal moves found.
    '''

    def recurse_minimax(board: List[int], turn: int, depth: int) -> Tuple[float, Tuple[int, Optional[bool]]]:
        '''
        Recursively iterates through all legal moves and returns the best move and its CPS score.
        
        Always optimizes for the the best move of the player denoted by `turn`.
        
        `turn` alternates between each move.
        
        An example of the alg with depth = 2 (i.e. 2 half-turns):

        Say recurse is called with `turn` = NOUGHTS, depth = 2, then the first move will be made by NOUGHTS.
        A random move by NOUGHTS will be trialed and now recurse is called again as CROSSES' turn, depth 1.
        A random move by CROSSES will be trialed and now recurse is called again as NOUGHTS' turn, depth 0.
        
        However, at depth 0, NOUGHTS doesn't get to make a random move, instead the CPS score is returned
        as the base case of the recursion. This score represents the state of the board after NOUGHTS has
        made one move and CROSSES has made one move.
        
        Back up at depth 1, it was CROSSES's turn, and the function tries to optimize the best move for CROSSES.
        
        It tries all possible moves such CROSSES will make the best move given the random initial move made by
        NOUGHTS, and returns the CPS score after such a play has been made.
        
        Back up at depth 2, it was NOUGHTS's turn, and the function keeps trying other moves
        for NOUGHTS such that the best play CROSSES will make after such a move will be maximally in 
        favour for NOUGHTS and minimally in favour of CROSSES.

        ### Parameters

        - `board`: The current state of the board
        - `turn`: Which player to find the best move for.
        - `depth`: How many moves ahead to search (1 represents next move, terminate recursion)

        ### Returns

        The best move for the current player, in the form `(cps: float, move: (column: int, pop: bool))`
        '''

        assert depth >= 0, "Depth must be >= 0"

        # If current board state has an immediate winning move this current turn
        # return it.

        winning_move = find_immediate_win(board, turn)

        if winning_move is not None:
            # If there are winning moves, return the first one
            # 10000 is equivalent to 10 immediate win opportunities given a free half-turn move.
            # (symbolizes a definite win in this state)
            score = 10000 if turn == NOUGHTS else -10000
            return score, winning_move

        # Iterate legal moves and recurse each scenario

        if depth == 0:
            # No deeper recursion, return CPS score
            return eval_cps(board), None

        best_move = None # if no legal moves, will return stalemate.
        
        # best score in favour of Nought is maximal
        # best score in favour of Cross is minimal
        # initialize best_score to be the opposite end of optimal score depending on 
        # which player to optimize for.
        best_score = -100000 if turn == NOUGHTS else 100000

        for col in range(COLS):
            for pop in [True, False]:
                if not check_move(board, turn, col, pop):
                    continue

                board_copy = apply_move(board, turn, col, pop)
                score, _ = recurse_minimax(board_copy, next_player(turn), depth - 1)

                if turn == NOUGHTS:
                    if score > best_score:
                        best_score = score
                        best_move = (col, pop)
                elif turn == CROSSES:
                    if score < best_score:
                        best_score = score
                        best_move = (col, pop)

        return best_score, best_move

    _, best_move = recurse_minimax(board, turn, depth)
    
    return best_move


def check_board_empty(board: List[int]) -> bool:
    '''
    Checks if the board is empty (no pieces on the board)

    ### Parameters

    - `board`: The current state of the board

    ### Returns

    True if the board is empty, False otherwise.
    '''
    return all(piece == 0 for piece in board)


def computer_move(board: List[int], turn: int, level: int) -> Tuple[int, bool]:
    '''
    Evaluates the 'best' move to make for a given `turn` (i.e. player), depending on `level` of
    difficulty and `board` state.

    ### Parameters
        - `board`: the board state
        - `turn`: the player number of which the computer is supposed to make a move for.
        - `level`: the difficulty level of the computer.

    ### Returns
        A tuple of the form `(col, pop)`, where `col` is the column to drop/pop the piece,
        and `pop` is `True` if pop, `False` if drop.

    ### Raises
        - `AssertionError` if `level` is not within 1-4.
        - `RuntimeError` if the computer cannot find a legal move to make (unhandled stalemate)
    '''

    assert 1 <= level <= 4, "Invalid computer level"
    assert 1 <= turn <= 2, "Unsupported player number"


    # ==========================================
    #          LEVEL 1: Random move
    # ==========================================

    if level == 1:
        # Trivial. Just make any legal random move.
        cols = [col for col in range(COLS)]
        random.shuffle(cols)

        for col in cols:
            # This code is horribly inperformant but whatever.
            can_drop = check_move(board, turn, col, False)
            can_pop = check_move(board, turn, col, True)
            if can_drop and can_pop:
                return col, random.choice([True, False])
            elif can_drop:
                return col, False
            elif can_pop:
                return col, True
        
        # If code reached here, then it is a stalemate. However, this should never happen.
        # Stalemate should be checked before any move is made by any player or computer.

        raise RuntimeError("Unhandled stalemate. Computer has no moves.")

    # ==========================================
    #     LEVEL 2: Immediate win/no loss
    # ==========================================
    elif level == 2:
        # Somewhat trivial, just make a move that doesn't directly allow the opponent to win the game, and wins
        # if an almost-win board state is reached. (Use brute-force)

        # Brute force all legal moves (up to 14 of them only) to see if any of them immediately wins:

        if (winning_move := find_immediate_win(board, turn)) is not None:
            return winning_move
        
        # Otherwise, make any random move that doesn't allow opponent to immediately win.

        cols = [col for col in range(COLS)]
        random.shuffle(cols)

        # Keeps track of the last legal move found:
        last_legal_move = None

        for col in cols:
            can_drop = check_move(board, turn, col, False)
            can_pop = check_move(board, turn, col, True)

            if can_drop:
                # NOTE: This naive 'AI' will prefer making drops over pops.

                last_legal_move = (col, False)

                # check that a drop move won't result in immediate win for opponent.
                board_copy = apply_move(board, turn, col, False)
                if find_immediate_win(board_copy, next_player(turn)) is None:
                    # If no immediate win for opponent, then this move is fine. Return it.
                    return col, False
            
            if can_pop:
                last_legal_move = (col, True)

                # check that a pop move won't result in immediate win for opponent.
                board_copy = apply_move(board, turn, col, True)
                if find_immediate_win(board_copy, next_player(turn)) is None:
                    # If no immediate win for opponent, then this move is fine. Return it
                    return col, True
        
        if last_legal_move is not None:
            # If code reached here, then the computer is zugzwanged. Return the last legal move found.
            # Admit defeat.
            return last_legal_move
        
        # If code reached here, it is an uncaught stalemate.
        raise RuntimeError("Unhandled stalemate. Computer has no moves.")

    # ==========================================
    #        LEVEL 3: CPS heuristic
    # ==========================================
    elif level == 3:
        # (Optional) Use the CPS metric. Choose the move that maximizes CPS for the computer player.

        if check_board_empty(board):
            # If board is empty, always play drop center column, that is always the best opening move.
            return (COLS - 1) // 2, False

        best_move_so_far = None
        best_score_so_far = -math.inf

        for col in cols:
            can_drop = check_move(board, turn, col, False)
            can_pop = check_move(board, turn, col, True)

            if can_drop:
                # check that a drop move won't result in immediate win for opponent.
                board_copy = apply_move(board, turn, col, False)
                if (cps := eval_cps(board_copy)) > best_score_so_far:
                    best_score_so_far = cps
                    best_move_so_far = (col, False)
            
            if can_pop:
                board_copy = apply_move(board, turn, col, True)
                if (cps := eval_cps(board_copy)) > best_score_so_far:
                    best_score_so_far = cps
                    best_move_so_far = (col, True)
        
        if best_move_so_far is not None:
            return best_move_so_far

        # If code reached here, it is an uncaught stalemate.
        raise RuntimeError("Unhandled stalemate. Computer has no moves.")
    elif level >= 4:
        # (Optional) Use min-max. Use CPS metric as scoring system for min-max algorithm.
        # min-max depth = level - 2
        # level 4 = depth 2
        # level 5 = depth 3
        # level 6 = depth 4
        # etc...
        
        if check_board_empty(board):
            # If board is empty, always play drop center column, that is always the best opening move.
            return (COLS - 1) // 2, False
        
        best_move = decide_minimax_move(board, turn, level - 2)
        
        if best_move is not None:
            return best_move
        
        raise RuntimeError("Unhandled stalemate. Computer has no moves.")
                
    return (0,False)


def display_board(board: List[int]):
    '''
    Takes in the board state and displays it by any means.
    '''
    num_rows = len(board) // COLS
    file_index = 1000 # trailing zeroes required as windows sort 111 as 'smaller than' 2 even though 111 > 2

    # Clear the meme board
    for fname in os.listdir('epic_board'):
        path = os.path.join('epic_board', fname)
        os.remove(path)
    
    print("col:  0  1  2  3  4  5  6")
    print()
    for row in range(num_rows - 1, -1, -1): # the 'first' row represents the bottom, so start from the 'last' row
        print("    ", end="")
        for col in range(COLS):
            # Trivial display: done in terminal
            # Meme display: done in file explorer in epic_board/ directory.
            #               pieces are displayed as image files.
            piece = board[row * COLS + col]
            if piece == 0:
                print("  .", end="")
                copyfile("imgs/blank.png", f"epic_board/{file_index}.png")
            elif piece == NOUGHTS:
                print("  O", end="")
                copyfile("imgs/yellow.png", f"epic_board/{file_index}.png")
            elif piece == CROSSES:
                print("  X", end="")
                copyfile("imgs/red.png", f"epic_board/{file_index}.png")
            
            file_index += 1

        print()


def test_computer_vs_computer(num_rows: int, comp1_level: int, comp2_level: int):
    '''
    Function to test computer player against computer player
    
    ### Parameters:
    
    - `num_rows`: number of rows in the board
    - `comp1_level`: level of the first computer player
    - `comp2_level`: level of the second computer player
    '''
    board = [0]*num_rows*7 # init new board
    
    print(f"Starting Computer (lvl {comp1_level}) vs Computer (lvl {comp2_level})")
    display_board(board)
    
    turn = 1 # player 1 starts
    
    while True:
        curr_time = time()
        
        if check_stalemate(board, turn):
            print(f"Player {turn} has no moves and is stalemated. Draw!")
            break
        
        move_col, move_pop = computer_move(board, turn, comp1_level if turn == 1 else comp2_level)
        apply_move(board, move_col, move_pop, turn)
        
        display_board(board)
        
        print()
        
        time_elapsed = time() - curr_time
        print(f'P{turn} thought for {time_elapsed:.2f} seconds\n')
        
        if check_victory(board, turn):
            print(f"Player {turn} wins!")
            break
        
        turn = next_player(turn)
        
        if time_elapsed < 1.5:
            # Make each move take at least 1.5 seconds so the game doesn't go by too fast
            sleep(1.5 - time_elapsed)


def menu():
    '''
    Game menu. 
    
    User to select between PvP or PvAI.

    -----
    
    If PvP, implementation is straightforward.

    1. `display_board()`

    2. Allow player 1 to make a move.

    3. Go through move-making subroutine
        - Sanitize the input, make sure column is not out of bounds, 
          and that a truthy/falseyy value is given for `pop`.
        - Check if move is valid using `check_move()`.
        - If neither of the above passed, ask for input again.
        - Retrieve the move to make
    
    4. Apply the obtained move using `apply_move()`

    5. Repeat the 1-4 for player 2, then alternate between the 2 players, 
      until `check_victory()` returns a non-zero value, or until `check_stalemate()` returns `True`.

    -----

    If PvAI, user is prompted to select difficulty level, 
    and choose whether or not player or computer goes first.

    1. `display_board()`
    2. Allow player/computer to make a move.
    3. If computer's turn, evaluate `computer_move()` to obtain the best move to make.
    4. If player's turn, go through move-making subroutine to obtain move from player.
    5. Apply the obtained move using `apply_move()`
    6. Repeat 1-5 until `check_victory()` returns a non-zero value, or `check_stalemate()` returns `True`.
    '''
    pass

if __name__ == "__main__":
    test_computer_vs_computer(6, 3, 3)
