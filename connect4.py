import os
import random
from shutil import copyfile
from time import sleep, time
from typing import Any, Callable, List, Optional, Tuple

'''
NOTE 1: The `board` list that represents board state will have 7 * c elements,
        where `c` is the number of columns.

        The first 7 elements in the list represent the **bottom** row of the board,
        the next 7 represent the row above it, and so on.

NOTE 2: Since all functions are immutable/have no side effects, this should be a
        purely functional solution, i.e. no global mutable variables are required.
'''


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

    return 1 if who_played == 2 else 2


def display_player(turn: int) -> str:
    '''
    Convert player number to human-readable display
    '''
    return "O (Yellow)" if turn == 1 else "X (Red)"


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
    assert 0 <= col < 7, 'Invalid column'
    
    col_pieces = board[col::7] # represents pieces of given `col` from bottom to top.
    
    # A move is valid if it is Drop (not pop) and the column is not full (topmost piece in column 0),
    return (not pop and col_pieces[-1] == 0) \
        or (pop and col_pieces[0] == turn) # or if it is Pop and the bottom piece belongs to the current player.


def check_stalemate(board: List[int], turn: int) -> bool:
    '''
    Checks if the given player is in a stalemate (has no legal moves left).

    This will only return `True` in the very rare case that the board is full AND
    all the pieces at the bottom are the opponent's pieces, not allowing the current
    player to pop or drop anything. 
    
    This is an improbable case and should never happen,
    but its nice to leave this here for future-proofing in case a non-pop version of
    connect 4 is implemented.

    ### Parameters

    - `board`: The current state of the board
    - `turn`: Which player to check for stalemate

    ### Returns

    `True` if current `turn` player has no legal moves left. `False` otherwise.
    '''
    for col in range(7):
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
        board_copy[row * 7 + col] = turn
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
    num_rows = len(board) // 7

    # Check horizontals (left to right)
    for row in range(num_rows):
        # Check if there are 4 consecutive pieces of the same color
        # in a row.

        streak_piece = 0 # the current player number which has N pieces in a row. 0 means no player.
        index_of_streak_start = 0 # the beginning index of the N pieces in a row

        for col in range(7):
            piece = board[row * 7 + col]
            if piece != streak_piece: # streak is broken
                if streak_piece != 0 and col - index_of_streak_start >= 4:
                    # 4 or more consecutive pieces = win
                    if streak_piece == 1:
                        noughts_wins = True
                    elif streak_piece == 2: # redundant elif, but here for future-proofing multiplayers if needed.
                        crosses_wins = True
                index_of_streak_start = col # reset streak index
                streak_piece = piece
        
        # Check if row ended with a winning streak:
        if streak_piece != 0 and 7 - index_of_streak_start >= 4:
            if streak_piece == 1:
                noughts_wins = True
            elif streak_piece == 2:
                crosses_wins = True
    
    # Checking verticals and diagonals only make sense if num_rows >= 4
    if num_rows >= 4:
        # Check verticals (bottom to top)
        for col in range(7):
            # Check if there are 4 consecutive pieces of the same color
            # in a column.

            streak_piece = 0
            index_of_streak_start = 0

            for row in range(num_rows):
                piece = board[row * 7 + col]
                if piece != streak_piece:
                    if streak_piece != 0 and row - index_of_streak_start >= 4:
                        if streak_piece == 1:
                            noughts_wins = True
                        elif streak_piece == 2:
                            crosses_wins = True
                    index_of_streak_start = row
                    streak_piece = piece

            # Check if end of column has a winning streak:
            if streak_piece != 0 and num_rows - index_of_streak_start >= 4:
                if streak_piece == 1:
                    noughts_wins = True
                elif streak_piece == 2:
                    crosses_wins = True

        # Check up-left diagonals (bottom-right to top-left)
        
        # contains all starting bottom-right points such that diagonals have at least
        # 4 pieces in them.
        starting_coords = [(0, x) for x in range(3, 7)]
        starting_coords += [(x, 7 - 1) for x in range(1, num_rows - 3)]

        # traverse one diagonal at a time from the above starting points
        for row, col in starting_coords:
            streak_piece = 0
            index_of_streak_start = 0
            diagonal_idx = 0 # The (n+1)th piece of the current diagonal

            while row + diagonal_idx < num_rows and col - diagonal_idx >= 0:
                piece = board[(row + diagonal_idx) * 7 + col - diagonal_idx]
                if piece != streak_piece:
                    if streak_piece != 0 and diagonal_idx - index_of_streak_start >= 4:
                        if streak_piece == 1:
                            noughts_wins = True
                        elif streak_piece == 2:
                            crosses_wins = True
                    index_of_streak_start = diagonal_idx
                    streak_piece = piece
                diagonal_idx += 1

            # Check if the last few pieces are a winning streak:
            if streak_piece != 0 and diagonal_idx - index_of_streak_start >= 4:
                if streak_piece == 1:
                    noughts_wins = True
                elif streak_piece == 2:
                    crosses_wins = True

        # Check up-right diagonals (bottom-left to top-right)

        # similar to above, contains all starting bottom-left points such that diagonals have at least
        # 4 pieces in them.

        starting_coords = [(0, x) for x in range(7 - 4, -1, -1)]
        starting_coords += [(x, 0) for x in range(1, num_rows - 3)]

        for row, col in starting_coords:
            streak_piece = 0
            index_of_streak_start = 0
            diagonal_idx = 0

            while row + diagonal_idx < num_rows and col + diagonal_idx < 7:
                piece = board[(row + diagonal_idx) * 7 + col + diagonal_idx]
                if piece != streak_piece:
                    if streak_piece != 0 and diagonal_idx - index_of_streak_start >= 4:
                        if streak_piece == 1:
                            noughts_wins = True
                        elif streak_piece == 2:
                            crosses_wins = True
                    index_of_streak_start = diagonal_idx
                    streak_piece = piece
                diagonal_idx += 1

            # Check if the last few pieces are a winning streak:
            if streak_piece != 0 and diagonal_idx - index_of_streak_start >= 4:
                if streak_piece == 1:
                    noughts_wins = True
                elif streak_piece == 2:
                    crosses_wins = True

    if noughts_wins and crosses_wins:
        return next_player(who_played)
    elif noughts_wins:
        return 1
    elif crosses_wins:
        return 2
    
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
    for col in range(7):
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

    for col in range(7):
        for pop in [True, False]:
            if not check_move(board, turn, col, pop):
                continue

            board_copy = apply_move(board, turn, col, pop)

            if check_victory(board_copy, turn) == turn:
                # Found a winning move, append it to the list
                winning_moves.append((col, pop))
    
    # No winning moves
    return winning_moves


def eval_cps(board: List[int], depth: int = 3) -> float:
    '''
    Evaluates the CPS score metric (see README.md)
    
    This metric is stateless and doesn't depend on whose turn it is
    currently. It measures the winning opportunities of each player 
    (Noughts winning will contribute a positive score, crosses winning
    will contribute a negative score)

    ### Parameters

    - `board`: The current state of the board
    - `depth`: How many free-moves (half-turns without opponent being able to move)
               to look ahead. (default: 3, but use 1 or 2 when using this scoring system with minmax)
               
               NOTE: increasing depth by 1 will increase the expected variation of score by 10 fold.
               Do not compare scores evaluated using differing depths!
    
    ### Returns
    
    A float score that is greater (+ve) if NOUGHTS have more winning opportunities,
    and smaller (-ve) if CROSSES have more winning opportunities.
    
    A score of +1000 represents Nought is able to win in 1 free half-turn.
    -100 means that Crosses can win in 2 free half-turns
    900 means that Noughts can win in 1 free half-turn and Crosses can win in 2 free half-turns.
    
    10000 means Nought has already won.
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
        
        # Represents how many CPS points each immediate win is worth
        # at this current depth.
        immediate_win_worth = 10 ** (3 - depth + depth_remaining)
        
        # If already won position at initial depth, return 10000
        
        if check_victory(board, turn) == turn and depth_remaining == depth:
            return 10000
        
        # Find all immediately winning moves for current player

        winning_moves = find_immediate_win_multiple(board, turn)
        cps = immediate_win_worth * len(winning_moves)

        # Iterate legal moves and recurse each scenario

        if depth_remaining == 1:
            # No deeper recursion, return CPS score
            return cps

        for col in range(7):
            for pop in [True, False]:
                if not check_move(board, turn, col, pop):
                    continue

                board_copy = apply_move(board, turn, col, pop)
                cps += recurse_free_moves(board_copy, depth_remaining - 1, turn)

        return cps

    tally += recurse_free_moves(board, depth, 1)
    tally -= recurse_free_moves(board, depth, 2)

    return tally

def find_best_move(board: List[int], player: int, starting_depth: int) -> Tuple[float, float, float, Optional[Tuple[int, bool]]]:
    '''
    Finds the best move for computer using minmax with alpha-beta pruning.
    
    ### Parameters

    - `board`: The current state of the board
    - `turn`: Which player to find the best move for.
    - `depth_remaining`: How many half-turns deep to search (depth = 1 is the base case of the recursion)

    ### Returns

    `(score: float, alpha: float, beta: float, move: (column: int, pop: bool))`
    
    - `score`: the best score for the current player that can be attained from current board situation
    - `alpha`: the best score in favour of NOUGHTS that NOUGHTS can guarantee/force
    - `beta`: the best score in favour of CROSSES that CROSSES can guarantee/force
    - `move`: The best move for the current player this turn.
              `None` if stalemate or if current player will definitely lose.
    '''
    
    assert starting_depth > 0, "Starting depth must be greater than 0"
    
    def alphabeta_minmax(
            board: List[int], 
            turn: int, 
            depth_remaining: int, 
            alpha: float = -999999, 
            beta: float = +999999
        ) -> Tuple[float, float, float, Optional[Tuple[int, bool]]]:
        '''
        Minmax algorithm with alpha-beta pruning. 
        (Inspired by https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)

        ### Parameters

        - `board`: The current state of the board
        - `turn`: Which player to currently maximize the best move for.
        - `depth_remaining`: How many half-turns left to search (depth = 1 is the base case of the recursion)
        - `alpha`: The best score that NOUGHTS (maximizing player) can guarantee so far
        - `beta`: The best score that CROSSES (minimizing player) can guarantee so far

        ### Returns

        `(score: float, alpha: float, beta: float, move: (column: int, pop: bool))`
        
        - `score`: the best score for the current player that can be attained from current board situation
        - `alpha`: the best score that NOUGHTS can guarantee/force
        - `beta`: the best score that CROSSES can guarantee/force
        - `move`: The best move made by the current player this turn. `None` if stalemate and no legal moves found.
        
        NOTE: the returned alpha and beta aren't used recursively, they are only returned for the sake of debugging/statistics.
        '''

        # commented out to increase speed
        # assert depth_remaining >= 0, "Depth must be >= 0"

        # If current board state has an immediate winning move this current turn
        # return it.

        winning_move = find_immediate_win(board, turn)
        
        # Check terminal nodes (win cases)
        if winning_move is not None:
            # If there are winning moves, return the first one
            # (symbolizes a definite win in this state)
            score = 100000 if turn == 1 else -100000
            return score, (+100000 if turn == 1 else alpha), (-100000 if turn == 2 else beta), winning_move

        # Iterate legal moves and recurse each scenario

        if depth_remaining == 0:
            # No deeper recursion, return CPS score
            # Evaluate cps with lesser depth to speed up computation (since we're minmaxing recursively anyways)
            return eval_cps(board, 2), alpha, beta, None

        best_move = None # if no legal moves, will return stalemate.
        
        # stores the best score obtainable in favour of the current player.
        # best score in favour of Nought is maximal
        # best score in favour of Cross is minimal
        # initialize best_score to be the opposite end of optimal score depending on 
        # which player to optimize for.
        best_score = alpha if turn == 1 else beta
        
        for pop in [False, True]:
            cols = list(range(7))
            # shuffle cols so that AI doesn't prefer making moves on one side of the board over the other
            random.shuffle(cols)
            for col in cols:
                if not check_move(board, turn, col, pop):
                    continue

                board_copy = apply_move(board, turn, col, pop)
                
                # recursion step
                score, _, _, _ = alphabeta_minmax(board_copy, next_player(turn), depth_remaining - 1, alpha, beta)

                if turn == 1:
                    # maximizing player
                    
                    # update best score
                    if score > best_score:
                        best_score = score
                        best_move = (col, pop)
                    
                    if score >= beta:
                        # beta represents the 'best' (i.e. minimal) score that CROSSES can guarantee at this point in time
                        
                        # if the best score for NOUGHTS in this board state is better than the 
                        # best scenario CROSSES can force upon NOUGHTS,
                        # this position in the game would never be reached, since CROSSES would guarantee a better
                        # score by choosing a different move during their turn!
                        
                        # In other words, alpha must always be less than beta for a position to be plausible.
                        
                        # Thus, there is no need find any better moves for NOUGHTS, since they would
                        # never be attainable. We can terminate this branch and return any
                        # random 'best move' and 'best score', since they will never be used.
                        return best_score, alpha, beta, (col, pop)
                    
                    # if this board state is deemed plausible, update alpha
                    # alpha represents the 'best' (i.e. maximal) score that NOUGHTS can guarantee at this point in time
                    alpha = max(alpha, best_score)
                    
                elif turn == 2:
                    # minimizing player
                    if score < best_score:
                        best_score = score
                        best_move = (col, pop)
                    
                    if best_score <= alpha:
                        return best_score, alpha, beta, (col, pop)
                    
                    beta = min(beta, best_score)

        return best_score, alpha, beta, best_move
    
    return alphabeta_minmax(board, player, starting_depth)


def check_board_empty(board: List[int]) -> bool:
    '''
    Checks if the board is empty (no pieces on the board)

    ### Parameters

    - `board`: The current state of the board

    ### Returns

    True if the board is empty, False otherwise.
    '''
    return all(piece == 0 for piece in board)


def get_validated_input(
        prompt: str, 
        type_converter: Callable[[str], Any], 
        validator: Callable[[Any], bool], 
        type_error_msg: str, 
        validator_error_msg: Optional[str] = None) -> Any:
    '''
    A helper function to retrieve input data from user by prompt.
    
    
    This function repeatedly asks the user for input until a valid input is given,
    checking for both type conversion errors and validation errors.
    
    ### Parameters
    
    - `prompt`: The prompt to display to the user.
    - `type_converter`: A function that takes in a string and converts it to the desired type.
    - `validator`: A function that takes in the the type-converted input and returns `True`
                   if the input is valid, `False` otherwise.
    - `type_error_msg`: The message to display to the user if a type conversion error occurs.
    - `validator_error_msg`: (Optional) The message to display to the user if a validation error occurs.
                             If not specified, reverts to `type_error_msg`.
    
    ### Returns
    
    Returns a validated value of the desired type retrieved from user input.
    '''
    
    validator_error_msg = validator_error_msg or type_error_msg
    
    while True:
        try:
            user_input = input(prompt).strip() # automatically removes leading/trailing whitespace
            converted_input = type_converter(user_input)
            
            if validator(converted_input):
                return converted_input
            else:
                print(validator_error_msg)
        except ValueError:
            print(type_error_msg)


def player_move(board: List[int], turn: int) -> Tuple[int, bool]:
    '''
    Obtains move from player input, after confirming validity of move, and double-confirming with the player.
    
    This function will repeatedly request for moves until a valid move is provided.
    
    If the player selected a column where only either pop or drop is possible,
    this function will assume that the player intended to play the legal move and
    will automatically select pop/drop for the user based on what the legal move was.

    ### Parameters

    - `board`: The current board state.
    - `turn`: The current player

    ### Returns

    `(col: int, pop: bool)`: The move made by the player
    '''
    while True:
        # represents the 0-indexed column number of the move
        col = get_validated_input(
            f"{display_player(turn)}: enter column (1-{7}): ",
            lambda x: int(x) - 1, # convert from 1-based to 0-based
            lambda x: 0 <= x < 7,
            f"Please enter a number from 1 to {7}",
        )
        
        can_pop = check_move(board, turn, col, True)
        can_drop = check_move(board, turn, col, False)
        
        if can_pop and can_drop:
            # if both pop and drop are legal moves on this column, check with the user
            # which move was intended.
            pop_or_drop = get_validated_input(
                "Which move to make? (pop/drop/p/d)",
                lambda x: x.lower(),
                lambda x: x in ['pop', 'drop', 'p', 'd'],
                "Please enter either 'pop'/'p' or 'drop'/'d'",
            )
        
            pop = pop_or_drop.lower() in ['pop', 'p']
            
        elif can_pop:
            pop = True
        elif can_drop:
            pop = False
        else:
            print("You aren't able to make any valid moves on that column. Please choose another column.\n")
            continue
        
        # double-confirm move with player
        
        confirm_input = input(f"You are about to {'pop' if pop else 'drop'} your {display_player(turn)} piece on column {col + 1}.\n"
              "Confirm move? (<enter> to accept, 'n' to cancel): ").strip().lower()
        
        if confirm_input == "n":
            print("Move cancelled. Please choose another move.\n")
            continue
        
        return col, pop


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

    assert 1 <= level, "Invalid computer level"
    assert 1 <= turn <= 2, "Unsupported player number"


    # ==========================================
    #          LEVEL 1: Random move
    # ==========================================

    if level == 1:
        # Trivial. Just make any legal random move.
        cols = [col for col in range(7)]
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

        cols = [col for col in range(7)]
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
            return (7 - 1) // 2, False

        best_move_so_far = None
        best_score_so_far = -999999
        
        # if CROSSES' turn, optimize for lowest CPS possible
        # invert CPS score later ensure that maximum best score ==> minimum CPS ==> best move for CROSSES.
        turn_mult = 1 if turn == 1 else -1

        for col in range(7):
            can_drop = check_move(board, turn, col, False)
            can_pop = check_move(board, turn, col, True)

            if can_drop:
                # check that a drop move won't result in immediate win for opponent.
                board_copy = apply_move(board, turn, col, False)
                if (cps := turn_mult * eval_cps(board_copy)) > best_score_so_far:
                    best_score_so_far = cps
                    best_move_so_far = (col, False)
            
            if can_pop:
                board_copy = apply_move(board, turn, col, True)
                if (cps := turn_mult * eval_cps(board_copy)) > best_score_so_far:
                    best_score_so_far = cps
                    best_move_so_far = (col, True)
        
        if best_move_so_far is not None:
            return best_move_so_far

        # If code reached here, it is an uncaught stalemate.
        raise RuntimeError("Unhandled stalemate. Computer has no moves.")
    elif level >= 4:
        # (Optional) Use min-max. Use CPS metric as scoring system for min-max algorithm.
        # min-max depth = level - 2
        # etc...
        
        if check_board_empty(board):
            # If board is empty, always play drop center column, that is always the best opening move.
            return (7 - 1) // 2, False
        
        _, _, _, best_move = find_best_move(board, turn, level - 2)
        
        if best_move is not None:
            return best_move
        else:
            # there are no good moves to be made, computer is about to lose
            # just make a random move.
            return computer_move(board, turn, 1)
                
    return (0,False)


def display_board(board: List[int]):
    '''
    Takes in the board state and displays it by any means.
    '''
    num_rows = len(board) // 7
    file_index = 1000 # trailing zeroes required as windows sort 111 as 'smaller than' 2 even though 111 > 2

    # Clear the meme board
    for fname in os.listdir('epic_board'):
        try:
            path = os.path.join('epic_board', fname)
            os.remove(path)
        except PermissionError:
            # If the program doesn't have access rights to write to the directory
            # don't implement the file explorer board feature.
            pass
    
    
    def copyfile_if_perms(src, dst):
        try:
            copyfile(src, dst)
        except PermissionError:
            # If the program doesn't have access rights to write to the directory
            # don't implement the file explorer board feature.
            pass
    
    print("col:  1  2  3  4  5  6  7")
    print()
    for row in range(num_rows - 1, -1, -1): # the 'first' row represents the bottom, so start from the 'last' row
        print("    ", end="")
        for col in range(7):
            # Trivial display: done in terminal
            # Meme display: done in file explorer in epic_board/ directory.
            #               pieces are displayed as image files.
            piece = board[row * 7 + col]
            if piece == 0:
                print("  .", end="")
                copyfile_if_perms("imgs/blank.png", f"epic_board/{file_index}.png")
            elif piece == 1:
                print("  O", end="")
                copyfile_if_perms("imgs/yellow.png", f"epic_board/{file_index}.png")
            elif piece == 2:
                print("  X", end="")
                copyfile_if_perms("imgs/red.png", f"epic_board/{file_index}.png")
            
            file_index += 1

        print()


def display_move(col: int, pop: bool):
    '''
    Highlights which column a move was made in by printing
    ^ or v under the column.
    '''
    print(" " * (6 + col * 3), end="")
    print("v" if pop else "*")


def test_computer_vs_computer(num_rows: int, comp1_level: int, comp2_level: int, eval_depth: int = 4):
    '''
    Function to test computer player against computer player
    
    ### Parameters:
    
    - `num_rows`: number of rows in the board
    - `comp1_level`: level of the first computer player
    - `comp2_level`: level of the second computer player
    - `eval_depth`: depth of min-max used to evaluate board state
    '''
    board = [0]*num_rows*7 # init new board
    
    print(f"\nStarting Computer (lvl {comp1_level}) vs Computer (lvl {comp2_level})\n")
    
    sleep(0.5)
    
    display_board(board)
    
    sleep(1)
    
    turn = 1 # player 1 starts
    
    while True:
        curr_time = time()
        
        if check_stalemate(board, turn):
            print(f"Player {turn} (lvl {lvl}) has no moves and is stalemated. Draw!")
            break
        
        lvl = comp1_level if turn == 1 else comp2_level
        
        move_col, move_pop = computer_move(board, turn, lvl)
        board = apply_move(board, turn, move_col, move_pop)
        
        display_board(board)
        display_move(move_col, move_pop)
        
        print()
        
        time_elapsed = time() - curr_time
        print(f'P{turn} (lvl {lvl}) thought for {time_elapsed:.2f} seconds')
        
        if check_victory(board, turn):
            print(f"Player {turn} (lvl {lvl}) wins!")
            break
        
        # use high-depth computer to evaluate current position
        eval_score, alpha, beta, _ = find_best_move(board, turn, eval_depth)
        print(f'Eval score: {eval_score}, α: {alpha}, β: {beta}\n')
        
        turn = next_player(turn)
        
        time_elapsed = time() - curr_time
        
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
    
    print("Welcome to Connect 4!")
    print()
    print("You can choose to view the current board by opening the 'epic_board' folder in your file explorer/finder")
    print("Make sure you open the file explorer large enough so that each row has 7 columns.\n")
    
    # Main loop: contains game mode selection and game code.
    while True:
        
        game_mode: int = get_validated_input(
            "Select game mode:\n1. PvP\n2. PvAI\n3. AIvAI\n4. Exit\n",
            int,
            lambda x: 1 <= x <= 4,
            "Please enter a number from 1 to 4.",
        )
        
        if game_mode == 4:
            # Exit main loop.
            print("Thanks for playing!")
            break
        
        num_rows: int = get_validated_input(
            "How many rows should the board have? (1-10) ",
            int,
            lambda x: 1 <= x <= 10,
            "Please enter a number from 1 to 10"
        )

        # Create empty board.
        board: List[int] = [0]*num_rows*7
        
        # Player 1 starts first
        turn: int = 1
        
        if game_mode == 1:
            # PvP
            print("\nStarting Player vs Player\n")
            
            # Show empty board first
            display_board(board)
            
            # Game loop
            while True:
                if check_stalemate(board, turn):
                    print(f"{display_player(turn)} has no moves and is stalemated. Draw!")
                    break
                
                move_col, move_pop = player_move(board, turn)
                board = apply_move(board, turn, move_col, move_pop)
                
                display_board(board)
                display_move(move_col, move_pop)
                
                if check_victory(board, turn):
                    print(f"{display_player(turn)} wins!")
                    break
                
                turn = next_player(turn)
            
        elif game_mode == 2:
            # PvAI
            print("\nStarting Player vs Computer\n")
            
            # Represents which turn number is made by the human player.
            player_turn: int = get_validated_input(
                "Do you want to go first or second? (1/2)",
                int,
                lambda x: x in [1, 2],
                "Please enter either 1 or 2",
            )
            
            # Computer difficulty level.
            comp_lvl: int = get_validated_input(
                "Select computer difficulty:\n"
                "1. Very easy (random)\n"
                "2. Easy (naive)\n"
                "3. Intermediate (heuristic)\n"
                "4. Advanced (αβ minmax depth 2)\n"
                "5. Hard (αβ minmax depth 3)\n"
                "6. Very hard (αβ minmax depth 4)\n"
                "7. Impossible (αβ minmax depth 5)\n",
                int,
                lambda x: 1 <= x <= 7,
                "Please enter a number from 1 to 7.",
            )
            
            print('Game starting...')
            
            # Show empty board
            display_board(board)
            
            # Game loop
            while True:
                if check_stalemate(board, turn):
                    if turn == player_turn:
                        print(f"You have no moves left. Draw!")
                    else:
                        print(f"The computer has no moves left. Draw!")
                    break
                
                if turn == player_turn:
                    # Player's turn
                    move_col, move_pop = player_move(board, turn)
                else:
                    # Computer's turn
                    print("\n Computer is thinking...\n")
                    move_col, move_pop = computer_move(board, turn, comp_lvl)
                
                board = apply_move(board, turn, move_col, move_pop)
                
                display_board(board)
                
                if turn != player_turn:
                    display_move(move_col, move_pop)
                
                if check_victory(board, turn):
                    if turn == player_turn:
                        print(f"You won against lvl. {comp_lvl}!")
                    else:
                        print(f"You lost to lvl. {comp_lvl}!")
                    break
                
                turn = next_player(turn)
            
        elif game_mode == 3:
            # AIvAI
            
            print('\nStarting Computer vs Computer showcase\n')
            
            # Computer difficulty level.
            comp1_lvl: int = get_validated_input(
                "Select computer 1 difficulty:\n"
                "1. Very easy (random)\n"
                "2. Easy (naive)\n"
                "3. Intermediate (heuristic)\n"
                "4. Advanced (αβ minmax depth 2)\n"
                "5. Hard (αβ minmax depth 3)\n"
                "6. Very hard (αβ minmax depth 4)\n"
                "7. Impossible (αβ minmax depth 5)\n",
                int,
                lambda x: 1 <= x <= 7,
                "Please enter a number from 1 to 7.",
            )
            
            comp2_lvl: int = get_validated_input(
                "Select computer 2 difficulty:\n"
                "1. Very easy (random)\n"
                "2. Easy (naive)\n"
                "3. Intermediate (heuristic)\n"
                "4. Advanced (αβ minmax depth 2)\n"
                "5. Hard (αβ minmax depth 3)\n"
                "6. Very hard (αβ minmax depth 4)\n"
                "7. Impossible (αβ minmax depth 5)\n",
                int,
                lambda x: 1 <= x <= 7,
                "Please enter a number from 1 to 7.",
            )
            
            print('Game starting...')
            
            if comp1_lvl >= 5 or comp2_lvl >= 5:
                print("The AI levels are high... this may take a while.\n"
                      "The moves made may not make any intuitive sense\n"
                      "(e.g. AI may make a move that appears to be a free win, knowing it is already in a lost position)")
            
            # Show empty board
            display_board(board)
            
            test_computer_vs_computer(
                len(board)//7,
                comp1_lvl,
                comp2_lvl,
                eval_depth=max(comp1_lvl - 2, comp2_lvl - 2, 3) # Use evaluation depth of at least 3
            )
    

if __name__ == "__main__":
    menu()
