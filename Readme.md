# Counterfactual Regret Minimization Explained

## Overview

[Counterfactual Regret Minimization](http://modelai.gettysburg.edu/2013/cfr/cfr.pdf) (CFR) is an important machine learning algorithm for playing "imperfect information" games. These are games where some information about the state of the game is hidden from the players, but the rules and objectives are known. This is common, for example, in card games, where each player's cards are hidden from the other players. Thus, chess is a perfect information game (nothing is hidden), while Poker, Clue, Battleship, and Stratego are imperfect information games.

This repository is my attempt to explain CFR and some of its variations in a concise, simple way using code. As I was learning about CFR, I found some aspects difficult to understand, due to confusing terminology and implementations (in my opinion). I also found that the dense math of academic papers that introduced these algorithms didn't help much to explain them.

## Implementations

Each script in this repository demonstrates a particular variation of CFR in F#, a functional programming language for the .NET platform. For clarity, each script is self-contained – there is no code shared between them.

Functional programming means that these implementations contain no side-effects or mutable variables. I find that such code is easier to understand and reason about than the kind of Python typically used in machine learning. (I think the ML community could really benefit from better software engineering, but that's a topic for another day.) Implementing CFR this way also makes it much easier to parallelize.

## Kuhn Poker

These examples start with solutions for [Kuhn Poker](https://en.wikipedia.org/wiki/Kuhn_poker). Kuhn Poker is a good choice for explaining CFR because it is a simple imperfect information card game, but does not have an obvious "best" strategy.

At each decision point in Kuhn Poker, the active player always has a choice of two actions: bet/call is one action and check/fold is the other.

Note that Kuhn Poker is zero-sum, two player game. In other words, one player's gain is the other player's loss. For simplicity, the implementations of CFR contained here rely on this fact, and can be adapted to other zero-sum, two player imperfect information games as well.

## Regret

"Regret" is really a very confusing term, and "counterfactual regret" even more so. In CFR, we're actually trying to choose the actions that have the *highest* regret, meaning that we most regret not choosing them in the past. (Like I said, confusing.) An action's regret might be positive (good) or negative (bad).

I think it's much easier to conceptualize this as choosing actions that have the highest value, or utility, or advantage instead. In CFR, all of these terms mean roughly the same thing as regret.

For example, in Kuhn Poker, a player holding the Jack should never call a bet (info sets `Jb` and `Jcb`), because the opponent is guaranteed to win. The regret of calling in this situation is -2, because the player loses two points. (One for calling, and one for the ante.)

## Information sets

An information set ("info set") contains all of the active player's information about the state of the game at a given decision point. For example, in Kuhn Poker, info set `Qcb` describes the situation where Player 1 has a Queen (`Q`) and checked (`c`) as the first action, then Player 2 bet (`b`). Player 1 now has the option of calling or folding, but does not know whether Player 2 has the Jack or the King. There are 12 such info sets in Kuhn Poker:

| Player 1's turn | Player 2's turn |
| --------------- |---------------- |
| `J`             | `Jb`            |
| `Q`             | `Jc`            |
| `K`             | `Qb`            |
| `Jcb`           | `Qc`            |
| `Qcb`           | `Kb`            |
| `Kcb`           | `Kc`            |

The info sets in a game are the internal nodes of a directed acyclic graph. Each valid action at a decision point leads to either a child info set or a "terminal" state where the game is over.

## Regret matching

For each info set, we are going to track the total regret of each possible action over time. This is stored as a vector, indexed by action (index 0 for bet/call and index 1 for check/fold in Kuhn Poker):

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

The probability of reaching a particular info set is the product of the probability of each action leading to it. For example, in a game of alternating turns, the probability of reaching an info set might be 1/2 × 1/3 × 1/4 × 1/5 = 1/120, where 1/2 × 1/4 = 1/8 is Player 1's contribution to reaching this state and 1/3 × 1/5 = 1/15 is Player 2's contribution. Note that the overall reach probability of an info set is equal to the product of each player's contribution (1/8 × 1/15 = 1/120). In CFR, each player's contribution to the overall reach probability is tracked separately.

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

We can then recurse by appending the action to the history so far:

```fsharp
loop (history + action) reachProbs)
```

The result is a utility for each legal action. These utilities must be negated, since it's the opponent's turn to play in each child node. The utility of the info set can be then calculated by weighting the utility of each child node by its probability of being chosen:

```fsharp
let utility = actionUtilities * strategy
```

Lastly, we need to update the regrets and strategy for this info set. The regret for each action is the difference between its utility and the overall utility of the info set, weighted by the opponent's contribution to the reach probability of the info set:

```fsharp
let regrets =
    reachProbs[opponent] * (actionUtilities - utility)
```

The strategy is weighted by the current player's contribution to the reach probability:

```fsharp
let strategy =
    reachProbs[activePlayer] * strategy
```

The computed regrets and strategy are used to update the info set:

```fsharp
InformationSet.accumulate regrets strategy infoSet
```

The utility and the updated information sets (including its updated child info sets) are then returned to the caller.

## Training

In order to use CFR effectively, we have to apply it the initial state of the game many times. Each CFR application is called an "iteration". After each iteration, we can run CFR again using the updated info sets from the previous iteration, allowing the system to "learn" over time. Updates are incorporated by folding them into a map of info sets:

```fsharp
let infoSetMap =
    (infoSetMap, keyedInfoSets)
        ||> Seq.fold (fun acc (key, infoSet) ->
                Map.add key infoSet acc)
```

After all of the iterations are complete, we can compute both the average expected utility of the game for each player, as well as the average strategy for each player to use. These averaged results are guaranteed (in vanilla CFR) to converge on an optimal strategy ("Nash equilibrium") over time. We need to take the average (rather than using the results of the last iteration) because the raw strategies might "circle" around the optimum without ever reaching it. By taking the average, we find the strategy in the center of the circle.

In order to compute this average strategy, we must track the sum of the strategies computed during each CFR iteration (alongside the accumulated regrets):

```fsharp
type InformationSet =
    {
        ...

        /// Sum of strategies accumulated so far by this info set.
        StrategySum : Vector<float>
    }

module InformationSet =

    ...

    let getAverageStrategy infoSet =
        normalize infoSet.StrategySum
```

## Pruning

Vanilla CFR must visit every info set on each iteration, which makes it prohibitively expensive for games with very large game trees. One way to improve the performance of vanilla CFR is to prune irrelevant subtrees. For example, if the reach probability of an info set is zero for both players, then it will contribute nothing to the update for that iteration, so it can be ignored:

```fsharp
if Vector.forall ((=) 0.0) reachProbs then
    0.0, Array.empty        // prune
else
    loopNonTerminal history reachProbs
```

The results should be identical to vanilla CFR, except that it finishes faster. Unfortunately, this optimization doesn't speed up CFR for Kuhn Poker very much.

## [Monte Carlo sampling](https://proceedings.neurips.cc/paper_files/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf)

We can prune the game tree further, to speed up CFR even more. Instead of exploring every valid action at each info set, we choose a random subset of actions and ignore the others. (Using random samples in this way is called a "Monte Carlo" algorithm.) This introduces noise into CFR, so it takes more iterations to converge, but is still faster overall.

Two ways of using this approach in CFR are:

* Outcome sampling: At each info set, one action is chosen randomly for evaluation.
* External sampling: For each iteration, pick one of the players as the "updating" player (alternating between players each iteration). For the updating player, explore every action and update their info sets, as in vanilla CFR. For the opposing player, sample one action at random (with probabilities proportional to that info set's current strategy) and do not update their info sets.

To do this, pass the index of the updating player to the CFR function. Then, within the function, test whether the active player in this info set is the updating player. If not, pick a single action to evaluate:

```fsharp
let private cfr infoSetMap deal updatingPlayer =

    ...

        // get player's current strategy for this info set
    let strategy = InformationSet.getStrategy infoSet

    let utility, keyedInfoSets =

        if activePlayer = updatingPlayer then

            ... same as vanilla CFR

        else
                // sample a single action according to the strategy
            let action =
                Categorical.Sample(rng, strategy.ToArray())
                    |> Array.get KuhnPoker.actions
            let utility, keyedInfoSets =
                loop (history + action) reachProbs
            -utility, keyedInfoSets

    utility, keyedInfoSets
```

In the training loop, alternate the updating player on each iteration:

```fsharp
(Map.empty, Seq.indexed deals)
    ||> Seq.mapFold (fun infoSetMap (i, deal) ->
        let utility, keyedInfoSets =
            let updatingPlayer = i % KuhnPoker.numPlayers
            cfr infoSetMap deal updatingPlayer
```

Since we're only updating one player's info sets on each interaction, we have to make sure that each player sees an unbiased set of deals:

```fsharp
for _ = 1 to numIterations do
    yield permutations[rng.Next(permutations.Length)]   // avoid bias
```

Without this change, we'd have a problem, since the number of possible deals in Kuhn Poker is even. Player 0 would always update on deals 0, 2, and 4, while Player 1 would always update on deals 1, 3, and 5.

As with simple pruning, this optimization doesn't make much difference for a small game like Kuhn Poker, but can converge more quickly for a large game (as measured by elapsed time, not number of iterations).

## Leduc Hold'em

[Leduc Hold'em](https://arxiv.org/pdf/1207.1411) is another imperfect information poker game that makes a good subject for CFR. It is larger than Kuhn Poker, containing 288 info sets, instead of just 12, but is still relatively simple.

> In Leduc hold ’em, the deck consists of two suits with three cards in each suit. There are two rounds. In the first round a single private card is dealt to each player. In the second round a single board card is revealed. There is a two-bet maximum, with raise amounts of 2 and 4 in the first and second round, respectively. Both players start the first round with 1 already in the pot.

I've provided an implementation of vanilla CFR for Leduc Hold'em, to make it easy to compare with Kuhn Poker.

## Parallelization

Another way to speed up CFR is to evaluate a batch of iterations in parallel, then update the info sets at the end of the batch. This results in fewer, but larger, updates.

## Running the code

 To run a script, install .NET and then execute the script in F# Interactive via `dotnet fsi`. For example:

```
> dotnet fsi './Kuhn Poker/Vanilla CFR.fsx'
```

Expected output:

```
Running Kuhn Poker vanilla CFR for 50000 iterations

Average game value for first player: -0.05817

Strategy:
J  :    b: 0.22471, c: 0.77529
Jb :    b: 0.00003, c: 0.99997
Jc :    b: 0.33811, c: 0.66189
Jcb:    b: 0.00002, c: 0.99998
K  :    b: 0.65545, c: 0.34455
Kb :    b: 0.99997, c: 0.00003
Kc :    b: 0.99988, c: 0.00012
Kcb:    b: 0.99996, c: 0.00004
Q  :    b: 0.00014, c: 0.99986
Qb :    b: 0.33643, c: 0.66357
Qc :    b: 0.00023, c: 0.99977
Qcb:    b: 0.56420, c: 0.43580

Elapsed time: 00:00:00.5947431
```
