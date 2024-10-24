#r "nuget: MathNet.Numerics.FSharp"

open System

open MathNet.Numerics.Distributions
open MathNet.Numerics.LinearAlgebra

/// Kuhn poker
module KuhnPoker =

    /// Number of players.
    let numPlayers = 2

    /// Available player actions.
    let actions =
        [|
            "b"   // bet/call
            "c"   // check/fold
        |]

    /// Cards in the deck.
    let deck =
        [
            "J"   // Jack
            "Q"   // Queen
            "K"   // King
        ]

    /// Gets zero-based index of active player.
    let getActivePlayer (history : string) =
        history.Length % numPlayers

    /// Gets payoff for the active player if the game is over.
    let getPayoff (cards : string[]) = function

            // opponent folds - active player wins
        | "bc" | "cbc" -> Some 1

            // showdown
        | "cc" | "bb" | "cbb" as history ->
            let payoff =
                if history.Contains('b') then 2 else 1
            let activePlayer = getActivePlayer history
            let playerCard = cards[activePlayer]
            let opponentCard =
                cards[(activePlayer + 1) % numPlayers]
            match playerCard, opponentCard with
                | "K", _
                | _, "J" -> payoff   // active player wins
                | _ -> -payoff       // opponent wins
                |> Some

            // game not over
        | _ -> None

/// An information set is a set of nodes in a game tree that are
/// indistinguishable for a given player. This type gathers regrets
/// and strategies for an information set.
type InformationSet =
    {
        /// Sum of regrets accumulated so far by this info set.
        RegretSum : Vector<float>

        /// Sum of strategies accumulated so far by this info set.
        StrategySum : Vector<float>
    }

module InformationSet =

    /// Initial info set.
    let zero =
        let zero = DenseVector.zero KuhnPoker.actions.Length
        {
            RegretSum = zero
            StrategySum = zero
        }

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

    /// Accumulates results.
    let accumulate regrets strategy infoSet =
        {
            RegretSum = infoSet.RegretSum + regrets
            StrategySum = infoSet.StrategySum + strategy
        }

    /// Computes average strategy from accumulated strateges.
    let getAverageStrategy infoSet =
        normalize infoSet.StrategySum

module KuhnCfrTrainer =

    /// Random number generator.
    let private rng = Random(0)

    /// Obtains an info set corresponding to the given key.
    let private getInfoSet infoSetKey infoSetMap =
        infoSetMap
            |> Map.tryFind infoSetKey
            |> Option.defaultValue InformationSet.zero   // first visit

    /// Updates the active player's reach probability to reflect
    /// the probability of an action.
    let private updateReachProbabilities reachProbs activePlayer actionProb =
        reachProbs
            |> Vector.mapi (fun i x ->
                if i = activePlayer then
                    x * actionProb
                else x)

    /// Negates opponent's utilties (assuming a zero-zum game).
    let private getActiveUtilities utilities =
        utilities
            |> Seq.map (~-)
            |> DenseVector.ofSeq

    /// Evaluates the utility of the given deal via external
    /// sampling Monte Carlo counterfactual regret minimization.
    let private cfr infoSetMap deal updatingPlayer =

        /// Top-level CFR loop.
        let rec loop history reachProbs =
            match KuhnPoker.getPayoff deal history with
                | Some payoff ->
                    float payoff, Array.empty   // game is over
                | None ->
                    loopNonTerminal history reachProbs

        /// Recurses for non-terminal game state.
        and loopNonTerminal history reachProbs =

                // get info set for current state from this player's point of view
            let activePlayer = KuhnPoker.getActivePlayer history
            let infoSetKey = deal[activePlayer] + history
            let infoSet = getInfoSet infoSetKey infoSetMap

                // get player's current strategy for this info set
            let strategy = InformationSet.getStrategy infoSet

                // get utility of this info set
            if activePlayer = updatingPlayer then

                    // get utility of each action
                let actionUtilities, keyedInfoSets =
                    let utilities, keyedInfoSetArrays =
                        (KuhnPoker.actions, strategy.ToArray())
                            ||> Array.map2 (fun action actionProb ->
                                let reachProbs =
                                    updateReachProbabilities
                                        reachProbs
                                        activePlayer
                                        actionProb
                                loop (history + action) reachProbs)
                            |> Array.unzip
                    getActiveUtilities utilities,
                    Array.concat keyedInfoSetArrays

                    // utility of this info set is action utilities weighted by action probabilities
                let utility = actionUtilities * strategy

                    // accumulate updated regrets and strategy
                let keyedInfoSets =
                    let infoSet =
                        let regrets =
                            let opponent =
                                (activePlayer + 1) % KuhnPoker.numPlayers
                            reachProbs[opponent] * (actionUtilities - utility)
                        let strategy =
                            reachProbs[activePlayer] * strategy
                        InformationSet.accumulate regrets strategy infoSet
                    [|
                        yield! keyedInfoSets
                        yield infoSetKey, infoSet
                    |]

                utility, keyedInfoSets

            else
                    // sample a single action according to the strategy
                let action =
                    Categorical.Sample(rng, strategy.ToArray())
                        |> Array.get KuhnPoker.actions
                let utility, keyedInfoSets =
                    loop (history + action) reachProbs
                -utility, keyedInfoSets

        [| 1.0; 1.0 |]
            |> DenseVector.ofArray
            |> loop ""

    /// Trains for the given number of iterations.
    let train numIterations =

        let utilities, infoSetMap =

                // each iteration evaluates one possible deal
            let deals =
                let permutations =
                    [|
                        for card0 in KuhnPoker.deck do
                            for card1 in KuhnPoker.deck do
                                if card0 <> card1 then
                                    [| card0; card1 |]
                    |]
                seq {
                    for _ = 1 to numIterations do
                        yield permutations[rng.Next(permutations.Length)]   // avoid bias
                }

                // start with no known info sets
            (Map.empty, Seq.indexed deals)
                ||> Seq.mapFold (fun infoSetMap (i, deal) ->

                        // evaluate one game starting with this deal
                    let utility, keyedInfoSets =
                        let updatingPlayer = i % KuhnPoker.numPlayers
                        cfr infoSetMap deal updatingPlayer

                        // update info sets
                    let infoSetMap =
                        (infoSetMap, keyedInfoSets)
                            ||> Seq.fold (fun acc (key, infoSet) ->
                                    Map.add key infoSet acc)

                    utility, infoSetMap)

            // compute average utility per deal
        let utility =
            Seq.sum utilities / float numIterations
        utility, infoSetMap

let run () =

        // train
    let numIterations =
        if fsi.CommandLineArgs.Length > 1 then
            Int32.Parse(fsi.CommandLineArgs[1])
        else 500000
    printfn $"Running Kuhn Poker Monte Carlo CFR for {numIterations} iterations\n"
    let util, infoSetMap = KuhnCfrTrainer.train numIterations

        // expected overall utility
    printfn $"Average game value for first player: %0.5f{util}\n"
    assert(abs(util - -1.0/18.0) <= 0.02)

        // strategy
    printfn "Strategy:"
    for (KeyValue(key, infoSet)) in infoSetMap do
        let str =
            let strategy =
                InformationSet.getAverageStrategy infoSet
            (strategy.ToArray(), KuhnPoker.actions)
                ||> Array.map2 (fun prob action ->
                    sprintf "%s: %0.5f" action prob)
                |> String.concat ", "
        printfn $"%-3s{key}:    {str}"
    assert(
        let betAction =
            Array.IndexOf(KuhnPoker.actions, "b")
        let prob key =
            let strategy =
                infoSetMap[key]
                    |> InformationSet.getAverageStrategy
            strategy[betAction]
        let k = prob "K"
        let j = prob "J"
        j >= 0.0 && j <= 1.0/3.0            // bet frequency for a Jack should be between 0 and 1/3
            && abs((k / j) - 3.0) <= 0.1)   // bet frequency for a King should be three times a Jack

let timer = Diagnostics.Stopwatch.StartNew()
run ()
printfn ""
printfn $"Elapsed time: {timer}"
