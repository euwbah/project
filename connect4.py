import random
import math
from typing import List, Tuple

'''
NOTE 1: The `board` list that represents board state will have 7 * c elements,
        where `c` is the number of columns.

        The first 7 elements in the list represent the **bottom** row of the board,
        the next 7 represent the row above it, and so on.

NOTE 2: Since all functions are immutable/have no side effects, this should be a
        purely functional solution, i.e. no global mutable variables are required.
'''

# Enumerating player color and number.
# Yellow goes first.
YELLOW = 1
RED = 2

def check_move(board: List[int], turn: int, col: int, pop: bool) -> bool:
    '''
    Checks if a certain move is valid given a `board` state. Returns whether or not move is valid.

    ### Arguments
        - `board`: the board state
        - `turn`: which player makes this move
        - `col`: the column to drop/pop the piece. This is zero-indexed (first col is 0).
        - `pop`: `True` if pop, `False` if drop
    
    ### Returns
        `True` if move is valid, `False` otherwise
    '''
    return True

def apply_move(board: List[int], turn: int, col: int, pop: bool) -> List[int]:
    '''
    Applies a given move to the board. Returns the new board state.

    This is an **immutable** function with NO SIDE EFFECTS, i.e. the list
    referred to by the `board` variable is not modified.

    ### Arguments
        - `board`: the board state
        - `turn`: which player makes this move
        - `col`: the column to drop/pop the piece. This is zero-indexed (first col is 0).
        - `pop`: `True` if pop, `False` if drop

    ### Returns
        The new board state (list of ints)
    '''
    return board.copy()

def check_victory(board: List[int], who_played: int) -> int:
    '''
    Checks if a player has won the game. Returns the player who won, or 0 if no one has won.

    ||----------------------------------------------------------------------------------||
    ||NOTE: According to telegram chat, if some player somehow makes a move such that   ||
    ||the board would be in a winning position for BOTH players (e.g. via a pop move),  ||
    ||the player that made the move LOSES.                                              ||
    ||----------------------------------------------------------------------------------||

    ### Arguments
        - `board`: the board state
        - `who_played`: the player who just made a move

    ### Returns
        The player number of the player who won, or 0 if no one has won.
        If the board is in a position that is winning for both players, then return
        the OPPONENT player. The player who just made such a move loses.

        I.e. you lose if you make a move that would win the game for your opponent, even
        if it is winning for yourself.
    '''
    return -1

def computer_move(board: List[int], turn: int, level: int) -> Tuple[int, bool]:
    '''
    Evaluates the 'best' move to make for a given `turn` (i.e. player), depending on `level` of
    difficulty and `board` state.

    ### Arguments
        - `board`: the board state
        - `turn`: the player number of which the computer is supposed to make a move for.
        - `level`: the difficulty level of the computer.
            - 1: Trivial. 

    ### Returns
        A tuple of the form `(col, pop)`, where `col` is the column to drop/pop the piece,
        and `pop` is `True` if pop, `False` if drop.
    '''
    return (0,False)
    
def display_board(board: List[int]):
    '''
    Takes in the board state and displays it by any means.
    '''
    pass

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
      until `check_victory()` returns a non-zero value.

    -----

    If PvAI, user is prompted to select difficulty level, 
    and choose whether or not player or computer goes first.

    1. `display_board()`
    2. Allow player/computer to make a move.
    3. If computer's turn, evaluate `computer_move()` to obtain the best move to make.
    4. If player's turn, go through move-making subroutine to obtain move from player.
    5. Apply the obtained move using `apply_move()`
    6. Repeat 1-5 until `check_victory()` returns a non-zero value.
    '''
    pass

if __name__ == "__main__":
    menu()




    
