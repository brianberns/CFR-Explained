# Counterfactual Regret Minimization Explained

## Overview

Counterfactual Regret Minimization (CFR) is an important machine learning algorithm for playing "imperfect information" games. These are games where some information about the state of the game is hidden from the players, but the rules and objectives are known. This is common, for example, in card games, where each player's cards are hidden from the other players. Thus, chess is a perfect information game (nothing is hidden), while Poker, Clue, Battleship, and Stratego are imperfect information games.

This is my attempt to explain CFR and its variations in a concise, simple way using code. As I was learning about CFR, I found some aspects difficult to understand, due to confusing terminology and poor implementations (in my opinion). The dense math of academic papers that these algorithms didn't help much either.

## Implementations

Each script in this repository demonstrates a particular variation of CFR in F#, a functional programming language for the .NET platform. For clarity, each script is self-contained - there is no code shared between them.

Functional programming means that these implementations contain no side-effects or mutable variables. I find that such code is much easier to understand and reason about than the Python typically used in machine learning.

## Kuhn poker

Like other introductions to CFR, I've used [Kuhn poker](https://en.wikipedia.org/wiki/Kuhn_poke) in these examples because it is a very simple imperfect information card game, but does not have an obvious "best" strategy. The rules are described below.

Note that Kuhn poker is zero-sum, two player game. In other words, one player's gain is the other player's loss. For simplicity, the implementations of CFR described here rely on this fact and can be adapted to other zero-sum, two player imperfet information games as well.

## Kuhn poker rules

**Deck and Setup:**
* The game uses a deck of only three cards: King, Queen, and Jack.
* Each player antes 1 chip (a blind bet) at the start of the game.

**Dealing:**
* Each player is dealt one card at random, face down, and the third card is set aside unseen.

**Gameplay:**
* Player 1 can either check or bet 1 chip.
* If Player 1 checks, Player 2 can check or bet 1 chip.
  * If both players check, the player with the higher card wins the pot of 2 chips.
  * If Player 2 bets, Player 1 can fold or call.
    * If Player 1 folds, Player 2 wins the pot of 3 chips.
    * If Player 1 calls, there is a showdown, and the player with the higher card wins the pot of 4 chips.
* If Player 1 bets, Player 2 can fold or call.
  * If Player 2 folds, Player 1 wins the pot of 3 chips.
  * If Player 2 calls, there is a showdown, and the player with the higher card wins the pot of 4 chips.

**Winning:**
* The player with the higher card wins the pot in a showdown.
* If a player folds, the other player wins the pot.

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

History    Bet      Pass
J  :    [0.21587; 0.78413]
K  :    [0.66842; 0.33158]
Q  :    [0.00012; 0.99988]
Jb :    [0.00003; 0.99997]
Jc :    [0.33573; 0.66427]
Kb :    [0.99997; 0.00003]
Kc :    [0.99990; 0.00010]
Qb :    [0.33712; 0.66288]
Qc :    [0.00019; 0.99981]
Jcb:    [0.00002; 0.99998]
Kcb:    [0.99996; 0.00004]
Qcb:    [0.55388; 0.44612]
```
