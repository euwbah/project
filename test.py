from connect4 import *

def test():
    
    # ***************** check_move ***************** #
    print()
    
    board = [0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0]
    if check_move(board, 1, 1, False): print("test check_move 1 - OK ! Allow drop in open column")
    else: print("test check_move 1 - Problem in the check_move function output !")
    
    board = [0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0]
    if not check_move(board, 1, 1, True): print("test check_move 2 - OK ! Disallow pop with no piece at bottom.")
    else: print("test check_move 2 - Problem in the check_move function output !")
    
    board = [1,0,0,0,0,0,0,  2,0,0,0,0,0,0,  1,0,0,0,0,0,0,  2,0,0,0,0,0,0,  1,0,0,0,0,0,0,  2,0,0,0,0,0,0]
    if not check_move(board, 1, 0, False): print("test check_move 3 - OK ! Disallow drop in full column")
    else: print("test check_move 3 - Problem in the check_move function output !")
    
    board = [1,0,0,0,0,0,0,  2,0,0,0,0,0,0,  1,0,0,0,0,0,0,  2,0,0,0,0,0,0,  1,0,0,0,0,0,0,  2,0,0,0,0,0,0]
    if check_move(board, 1, 0, True): print("test check_move 4 - OK ! Allow pop column with self-owned piece at bottom")
    else: print("test check_move 4 - Problem in the check_move function output !")
    
    board = [1,0,0,0,0,0,0,  2,0,0,0,0,0,0,  1,0,0,0,0,0,0,  2,0,0,0,0,0,0,  1,0,0,0,0,0,0,  2,0,0,0,0,0,0]
    if not check_move(board, 2, 0, True): print("test check_move 5 - OK ! Disallow pop column with opponent piece at bottom")
    else: print("test check_move 5 - Problem in the check_move function output !")
    
    board = [0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0]
    if not check_move(board, 1, 1, True): print("test check_move 6 - OK ! Disallow pop with no piece at bottom (8 rows)")
    else: print("test check_move 6 - Problem in the check_move function output !")
    
    board = [1,0,0,0,0,0,0,  2,0,0,0,0,0,0,  1,0,0,0,0,0,0,  2,0,0,0,0,0,0]
    if not check_move(board, 1, 0, False): print("test check_move 7 - OK ! Disallow drop in full column (4 rows)")
    else: print("test check_move 7 - Problem in the check_move function output !")
   
    
    # ***************** apply_move ***************** #
    print()
    
    board = [1,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0]
    board_result = [1,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0]
    apply_move(board, 1, 0, False)
    if board == board_result: print("test apply_move 1 - OK ! Ensure apply_move has no side-effects")
    else: print("test apply_move 1 - Problem in the apply_move function output !")
    
    board = [0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0]
    board_result = [1,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0]
    board_tmp = apply_move(board, 1, 0, False)
    if board_tmp == board_result: print("test apply_move 2 - OK ! Test drop player 1 at col 0")
    else: print("test apply_move 2 - Problem in the apply_move function output !")
    
    board = [0,1,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0]
    board_result = [0,1,0,0,0,0,0,  0,2,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0]
    board_tmp = apply_move(board, 2, 1, False)
    if board_tmp == board_result: print("test apply_move 3 - OK ! Test drop player 2 on top of another piece")
    else: print("test apply_move 3 - Problem in the apply_move function output !")
    
    board = [1,1,0,0,0,0,0,  2,2,0,0,0,0,0,  2,0,0,0,0,0,0,  1,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0]
    board_result = [2,1,0,0,0,0,0,  2,2,0,0,0,0,0,  1,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0]
    board_tmp = apply_move(board, 1, 0, True)
    if board_tmp == board_result: print("test apply_move 4 - OK ! Test pop player 1 at col 0")
    else: print("test apply_move 4 - Problem in the apply_move function output !")
    
    board = [0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0]
    board_result = [1,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0]
    board_tmp = apply_move(board, 1, 0, False)
    if board_tmp == board_result: print("test apply_move 5 - OK ! Test drop player 1 (8 rows)")
    else: print("test apply_move 5 - Problem in the apply_move function output !")
    
    board = [1,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0]
    board_result = [1,0,0,0,0,0,0,  2,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0]
    board_tmp = apply_move(board, 2, 0, False)
    if board_tmp == board_result: print("test apply_move 6 - OK ! Test drop player 2 on top of another piece (4 rows)")
    else: print("test apply_move 6 - Problem in the apply_move function output !")
    
    
    # ***************** check_victory ***************** #
    print()
    
    # trivial case, no victory for anyone
    board = [1,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0]
    if check_victory(board, 1)==0: print("test check_victory 1 - OK ! No victory.")
    else: print("test check_victory 1 - Problem in the check_victory function output !")
    
    # player 1 just moved, victory for player 1
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 2 0 0 0 0 0 0
    # 2 0 0 0 0 0 0
    # 2 0 0 0 0 0 0
    # 1 1 1 1 0 0 0
    board = [1,1,1,1,0,0,0,  2,0,0,0,0,0,0,  2,0,0,0,0,0,0,  2,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0]
    if check_victory(board, 1)==1: print("test check_victory 2 - OK ! P1 Horizontal victory.")
    else: print("test check_victory 2 - Problem in the check_victory function output !")
    
    # player 2 just moved, victory for player 2
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 2 0 0 0 0 0 0
    # 2 0 0 0 0 0 0
    # 2 1 0 0 0 0 0
    # 2 1 1 1 0 0 0
    board = [2,1,1,1,0,0,0,  2,1,0,0,0,0,0,  2,0,0,0,0,0,0,  2,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0]
    if check_victory(board, 2)==2: print("test check_victory 3 - OK ! P2 Vertical victory")
    else: print("test check_victory 3 - Problem in the check_victory function output !")
    
    # player 1 just moved, victory for player 2
    # ----------------------------------------------------------------------------------
    # NOTE: According to telegram chat, if some player somehow makes a move such that  |
    # the board would be in a winning position for BOTH players (e.g. via a pop move), |
    # the player that made the move LOSES.                                             |
    # ----------------------------------------------------------------------------------
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 1 2 0 0 0 0 0
    # 1 2 0 0 0 0 0
    # 1 2 0 0 0 0 0
    # 1 2 0 0 0 0 0
    board = [1,2,0,0,0,0,0,  1,2,0,0,0,0,0,  1,2,0,0,0,0,0,  1,2,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0]
    if check_victory(board, 1)==2: print("test check_victory 4 - OK ! Both players vertical victory, win for opponent.")
    else: print("test check_victory 4 - Problem in the check_victory function output !")
    
    # player 1 just moved, diagonal victory for player 1
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 0 0 0 1 0 0 0
    # 0 0 1 2 0 0 0
    # 0 1 2 1 0 0 0
    # 1 2 2 2 0 0 0
    board = [1,2,2,2,0,0,0,  0,1,2,1,0,0,0,  0,0,1,2,0,0,0,  0,0,0,1,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0]
    if check_victory(board, 1)==1: print("test check_victory 5 - OK ! P1 bottom-left top-right diagonal victory")
    else: print("test check_victory 5 - Problem in the check_victory function output !")
    
    # player 1 just moved, P1 diagonal victory
    # 0 0 0 1 0 0 0
    # 0 0 0 2 1 0 0 
    # 0 0 0 2 2 1 0 
    # 0 0 0 1 1 2 1 
    # 0 0 0 2 2 1 1 
    # 0 0 0 1 2 2 2 
    # 0 0 0 1 1 2 1 
    # 0 0 0 2 1 1 1 
    board = [0,0,0,2,1,1,1,  0,0,0,1,1,2,1,  0,0,0,1,2,2,2,  0,0,0,2,2,1,1,  0,0,0,1,1,2,1,  0,0,0,2,2,1,0,  0,0,0,2,1,0,0,  0,0,0,1,0,0,0]
    if check_victory(board, 1)==1: print("test check_victory 6 - OK ! P1 top-left bottom-right diagonal victory (8 rows)")
    else: print("test check_victory 6 - Problem in the check_victory function output !")
    
    
    # ***************** computer_move ***************** #
    print()
    
    board = [1,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0]
    board_result = [1,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0]
    computer_move(board, 1, 1)
    computer_move(board, 1, 2)
    if board == board_result: print("test computer_move 1 - OK !")
    else: print("test computer_move 1 - Problem in the computer_move function output !")
    
    board = [1,1,1,0,0,0,0,  2,0,0,0,0,0,0,  2,0,0,0,0,0,0,  2,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0]
    if computer_move(board, 1, 2) in [(3, False)]: print("test computer_move 2 - OK !")
    else: print("test computer_move 2 - Problem in the computer_move function output !")
    
    board = [1,1,1,0,0,0,0,  2,0,0,0,0,0,0,  2,0,0,0,0,0,0,  2,0,0,0,0,0,0,  0,0,0,0,0,0,0]
    if computer_move(board, 1, 2) in [(3, False)]: print("test computer_move 3 - OK !")
    else: print("test computer_move 3 - Problem in the computer_move function output !")
  
   
test()