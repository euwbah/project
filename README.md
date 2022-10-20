# Connect 4 With Pop

## AI Strategy

### Connection Potential Score (CPS)

This is a heuristic scoring system to evaluate whether the state of the board is in favour of Noughts or Crosses. This scoring system is regardless of whose turn it is, and should be a stateless function.

The philosophy behind this function is such: a player is more likely to win if they have more opportunities to connect their pieces.

Hence, CPS should be a function of:

1. How close is one player to making a Connect-4?
2. How many distinct opportunities can such player make a Connect-4?

#### Let us take some cases in a 1x7 board:

It is Nought's turn to move, suppose the following board state:

1. `x x _ x _ _ _`

Here, Cross is in an almost-win configuration, and the only non-losing move is for Circle to drop on the 3rd column. We set the CPS to -1000 (heuristic) in favour of Crosses (since it is almost-winning), and no other move besides Nought drop 3 can neutralize the position.

2. `_ _ x x x _ _`

Here, Cross is also in an almost-win configuration, but there are **no ways** for Nought to neutralize the position. If Nought were to drop on 2nd or 5th column, Crosses can just drop on the opposite end to secure a win. CPS is maximally in favour of Crosses, more so than in the first scenario.

3. `x x _ _ _ _ _`

Here, Cross is 2 half-turns away from winning (assuming Nought doesn't get a turn). However, Nought can simply drop on 3/4 to neutralize the position. CPS is only slightly in favour of Crosses.

4. `_ _ x x _ _ _`

Cross is 2 half-turns away from winning, but even after Nought 2 or Nought 4, Cross still has winning chances (assuming Nought is playing without strategy). CPS is slightly in favour of Crosses, but more so than the above scenario 3.

5. `x x _ o _ _ _`

Here, even though Cross has 2 connected pieces, the position is not only neutral, but can be seen as winning for Nought (assuming Cross makes very naive plays). Assuming Nought does not drop their piece on column 4, there is no way for Cross at all to win, since there are no opportunities for Cross to connect 4. CPS is barely in favour of Noughts or neutral.

6. `_ _ _ o _ _ _`
   `o o o x o x x`
   `x x o o x o x`

Here, we have an almost-win state for Nought. Nought is allowed to drop 4th column since the 4th column has a `o` at the bottom, but Cross can prevent that by dropping 2nd column, which breaks the potential horizontal connection formed if Nought were to drop on 4th. This is not functionally different from the situation in Scenario 1 where one player is in an almost-win configuration and the opposition has only one or two moves to neutralize it. Hence, CPS here is 1000 in favour of Noughts.

#### Takeaway

We can see that having n connected pieces is not sufficient a metric to abstract out who is winning. Instead, it is a combination of having connected pieces and opportunities to take advantage of the connected pieces.

These opportunities can come either by means of **spaces** or **drops**.

Hence, CPS is meant to serve as a measure of opportunities of winning assuming both players independently can make up to 3 half-moves.

#### Implementation

