# Counterfactual Regret Minimization Explained

## Overview

Counterfactual Regret Minimization (CFR) is an important machine learning algorithm for playing "imperfect information" games. These are games where some information about the state of the game is hidden from the players, but the rules and objectives are known. This is common, for example, in card games, where each player's cards are hidden from the other players. Thus, chess is a perfect information game (nothing is hidden), while Poker, Clue, Battleship, and Stratego are imperfect information games.

This is my attempt to explain CFR and its variations in a concise, simple way using code. As I was learning about CFR, I found some aspects difficult to understand, due to confusing terminology and poor implementations (in my opinion). The dense math of academic papers that introduced these algorithms didn't help much either.

## Implementations

Each script in this repository demonstrates a particular variation of CFR in F#, a functional programming language for the .NET platform. For clarity, each script is self-contained – there is no code shared between them.

Functional programming means that these implementations contain no side-effects or mutable variables. I find that such code is much easier to understand and reason about than the kind of Python typically used in machine learning. (I think the ML community could really benefit from better software engineering, but that's a topic for another day.)

## Kuhn Poker

Like other introductions to CFR, I've used [Kuhn Poker](https://en.wikipedia.org/wiki/Kuhn_poke) in these examples because it is a very simple imperfect information card game, but does not have an obvious "best" strategy.

Note that Kuhn poker is zero-sum, two player game. In other words, one player's gain is the other player's loss. For simplicity, the implementations of CFR described here rely on this fact and can be adapted to other zero-sum, two player imperfect information games as well.

## Regret

"Regret" is really a very confusing term, and "counterfactual regret" even more so. In CFR, we're actually trying to choose the actions that have the *highest* regret, meaning that we most regret not choosing them in the past. (Like I said, confusing.)

I think it's much easier to conceptualize this as choosing actions that have the highest value, or utility, or "advantage" instead.

## Information sets

An information set contains all of the current player's information about the state of the game at a given decision point. For example, in Kuhn Poker, `Qcb` describes the situation where Player 1 has a Queen (`Q`) and checked (`c`) as the first action, then Player 2 bet (`b`). Player 1 now has the option of calling or folding, but does not know whether Player 2 has the Jack or the King. There are 12 such information sets in Kuhn Poker:

| Player 1's turn | Player 2's turn |
| --------------- |---------------- |
| `J`             | `Jb`            |
| `Q`             | `Jc`            |
| `K`             | `Qb`            |
| `Jcb`           | `Qc`            |
| `Qcb`           | `Kb`            |
| `Kcb`           | `Kc`            |

We use `c` to represent both a check and a fold, and we use `b` to represent both a bet and a call.

Note that the information sets in a game form a tree. Each valid action at a decision point leads to a child information set. Leaf nodes in the tree represent the final states of the game and are called "terminal" states.

## Running the code

 To run a script, install .NET and then execute the script in F# Interactive via `dotnet fsi`. For example:

```
> dotnet fsi '.\Vanilla CFR.fsx'
```

Expected output:

```
Running Kuhn Poker vanilla CFR for 10000 iterations

Expected average game value: -0.05556 (for first player)
Computed average game value: -0.05823

History   Bet    Check
J  :    0.21587 0.78413
K  :    0.66842 0.33158
Q  :    0.00012 0.99988
Jb :    0.00003 0.99997
Jc :    0.33573 0.66427
Kb :    0.99997 0.00003
Kc :    0.99990 0.00010
Qb :    0.33712 0.66288
Qc :    0.00019 0.99981
Jcb:    0.00002 0.99998
Kcb:    0.99996 0.00004
Qcb:    0.55388 0.44612
```
