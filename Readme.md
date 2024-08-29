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

"Regret" is really a very confusing term, and "counterfactual regret" even more so. In CFR, we're actually trying to choose the actions that have the *highest* regret, meaning that we most regret not choosing them in the past. (Like I said, confusing.) An action's regret might be positive (good) or negative (bad).

I think it's much easier to conceptualize this as choosing actions that have the highest value, or utility, or advantage instead. In CFR, all of these terms mean roughly the same thing as regret.

## Information sets

An information set ("info set") contains all of the current player's information about the state of the game at a given decision point. For example, in Kuhn Poker, `Qcb` describes the situation where Player 1 has a Queen (`Q`) and checked (`c`) as the first action, then Player 2 bet (`b`). Player 1 now has the option of calling or folding, but does not know whether Player 2 has the Jack or the King. There are 12 such info sets in Kuhn Poker:

| Player 1's turn | Player 2's turn |
| --------------- |---------------- |
| `J`             | `Jb`            |
| `Q`             | `Jc`            |
| `K`             | `Qb`            |
| `Jcb`           | `Qc`            |
| `Qcb`           | `Kb`            |
| `Kcb`           | `Kc`            |

We use `c` to represent both a check and a fold, and we use `b` to represent both a bet and a call.

The info sets in a game are the internal nodes of a tree. Each valid action at a decision point leads to either a child info set or a "terminal" state where the game is over.

## Regret matching

For each info set, we are going to track the total utility ("regret") of each possible action so far. This is stored as a vector, indexed by action (index 0 for bet and index 1 for call in Kuhn Poker):

```fsharp
type InformationSet =
    {
        /// Sum of regrets accumulated so far by this info set.
        RegretSum : Vector<float>

        ...
    }
```

To choose an action at a given decision point, we want to give each possible action a probability that is proportional to its utility, so that more useful actions are chosen more often. This is known as "regret matching".

One complication in regret matching is what to do about negative regrets (i.e. actions that have resulted in bad outcomes overall). Vanilla CFR clamps the regret of such actions to zero during regret matching, in order to prevent them from being chosen. If all of the info set's actions have a non-positive regret, vanilla CFR chooses one at random by assigning them all equal probability. Here is the regret matching implementation in F#:

```fsharp
module InformationSet =

    /// Uniform strategy: All actions have equal probability.
    let private uniformStrategy =
        DenseVector.create
            KuhnPoker.actions.Length
            (1.0 / float KuhnPoker.actions.Length)

    /// Normalizes a strategy such that its elements sum to
    /// 1.0 (to represent action probabilities).
    let private normalize strategy =

            // assume no negative values during normalization
        assert(Vector.forall (fun x -> x >= 0.0) strategy)

        let sum = Vector.sum strategy
        if sum > 0.0 then strategy / sum
        else uniformStrategy

    /// Computes regret-matching strategy from accumulated
    /// regrets.
    let getStrategy infoSet =
        infoSet.RegretSum
            |> Vector.map (max 0.0)   // clamp negative regrets
            |> normalize
```

The `getStrategy` function computes a "strategy" vector of action probabilities for the given info set.

## Reach probabilities

The probability of reaching a particular info set is the product of the probability of each action leading to it. For example, in a game of alternating turns, the probability of reaching an info set might be 1/2 × 1/3 × 1/4 × 1/5 = 1/120, where 1/2 × 1/4 = 1/8 is Player 1's contribution to reaching this state and 1/3 × 1/5 = 1/15 is Player 2's contribution. Note that the overall probability is equal to the product of each player's contribution (1/8 × 1/15 = 1/120). In CFR, each player's contribution to the overall reach probability is tracked separately.

In our implementation, the reach probabilities are stored in a vector that is indexed by player: index 0 for player 1, and index 1 for Player 2. At the start of a game, the reach probability vector is `[| 1.0; 1.0 |]`, since neither player has made a decision yet.

## Vanilla CFR

In the original "vanilla" version of CFR, the utility of an info set is determined by recursively evaluating the outcome of each possible action. First we obtain the current strategy vector for the info set via regret matching:

```fsharp
let strategy = InformationSet.getStrategy infoSet
```

For each action, the active player's reach probability must first be updated to reflect the likelihood of taking the action with the current strategy:

```fsharp
let reachProbs' =
    reachProbs
        |> Vector.mapi (fun i x ->
            if i = activePlayer then
                x * actionProb
            else x)
```

The result is a utility for each child action. From this, the utility of the info set can be calculated by weighting each child utility by its probability:

```fsharp
let utility = actionUtilities * strategy
```

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
