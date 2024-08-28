# Counterfactual Regret Minimization Explained

## Overview

Counterfactual regret minimization (CFR) is an important machine learning algorithm for playing "imperfect information" games. These are games where some information about the state of the game is hidden from the players, but the rules and objectives are known. This is common, for example, in card games, where each player's cards are hidden from the other players. Thus, chess is a perfect information game (nothing is hidden), while Poker, Clue, Battleship, and Stratego are imperfect information games.

What follows is my attempt to explain CFR and its variations in a simple and direct way using code. As I was learning about CFR, I found some aspects difficult to understand, due to (IMHO) confusing terminology and poor implementations. The dense math of academic papers that describe CFR didn't help any.

## Running the code

Each script in this repository demonstrates a variation of CFR. For clarity, each script is self-contained - there is no code shared between them. To run a script, install .NET and then execute the script in F# Interactive via `dotnet fsi`. For example:

```
> dotnet fsi '.\Vanilla CFR.fsx'
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

Why F#?