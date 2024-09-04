#r "nuget: MathNet.Numerics.FSharp"

open System
open MathNet.Numerics.LinearAlgebra

module String =

    let tryLast (str : string) =
        if str.Length = 0 then None
        else Some str[str.Length - 1]

module List =

    // http://stackoverflow.com/questions/286427/calculating-permutations-in-f
    let rec permutations = function
        | []      -> seq [List.empty]
        | x :: xs -> Seq.collect (insertions x) (permutations xs)
    and insertions x = function
        | []             -> [[x]]
        | (y :: ys) as xs -> (x::xs)::(List.map (fun x -> y::x) (insertions x ys))

/// Leduc hold'em.
// https://github.com/scfenton6/leduc-cfr-poker-bot
module LeducHoldem =

    /// Number of players.
    let numPlayers = 2

    /// Cards in the deck.
    let deck =
        [
            "J"; "J"   // Jack
            "Q"; "Q"   // Queen
            "K"; "K"   // King
        ]

    (*
     * Actions:
     *    x: check
     *    f: fold
     *    c: call
     *    b: bet
     *    r: raise
     *    d: community card dealt
     *)

    /// Action strings that end a round, without necessarily
    /// ending the game.
    let isRoundEnd = function
        | "bc"
        | "xx"
        | "xbc"
        | "brc"
        | "xbrc" -> true
        | _ -> false

    let isTerminal rounds =
        let round = Array.last rounds
        match String.tryLast round, rounds.Length with
            | Some 'f', _ -> true
            | _, 2 -> isRoundEnd round
            | _ -> false

    /// Gets legal actions for active player.
    let getLegalActions history =
        match String.tryLast history with
            | None
            | Some 'd'
            | Some 'x' -> [| "x"; "b" |]
            | Some 'b' -> [| "f"; "c"; "r" |]
            | Some 'r' -> [| "f"; "c" |]
            | _ -> failwith "Unexpected"

    let private ante = 1

    let private payoffs =
        Map [
            "xx", 0
            "bf", 0
            "xbf", 0
            "brf", 2
            "xbrf", 2
            "bc", 2
            "xbc", 2
            "brc", 4
            "xbrc", 4
        ]

    let private rank = function
        | "J" -> 11
        | "Q" -> 12
        | "K" -> 13
        | _ -> failwith "Unexpected"

    /// Gets payoff for the active player if the game is over.
    let getPayoff
        (playerCards : string[])
        communityCard
        (rounds : string[]) =

        if rounds.Length = 2 then
            let pot = ante + payoffs[rounds[0]] + 2 * payoffs[rounds[1]]
            match String.tryLast rounds[1] with
                | Some 'f' -> pot
                | _ ->   // showdown
                    let activePlayer = rounds[1].Length % numPlayers
                    let opponent = (activePlayer + 1) % numPlayers
                    if playerCards[activePlayer] = communityCard then
                        pot
                    elif playerCards[opponent] = communityCard then
                        -pot
                    else
                        let diff =
                            rank playerCards[activePlayer]
                                - rank playerCards[opponent]
                        if diff > 0 then pot
                        elif diff = 0 then 0
                        else -pot
        else
            assert(rounds.Length = 1)
            assert(String.tryLast rounds[0] = Some 'f')
            ante + payoffs[rounds[0]]

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
    let zero numActions =
        let zero = DenseVector.zero numActions
        {
            RegretSum = zero
            StrategySum = zero
        }

    /// Uniform strategy: All actions have equal probability.
    let private uniformStrategy numActions =
        DenseVector.create
            numActions
            (1.0 / float numActions)

    /// Normalizes a strategy such that its elements sum to
    /// 1.0 (to represent action probabilities).
    let private normalize strategy =

            // assume no negative values during normalization
        assert(Vector.forall (fun x -> x >= 0.0) strategy)

        let sum = Vector.sum strategy
        if sum > 0.0 then strategy / sum
        else uniformStrategy strategy.Count

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

module LeducCfrTrainer =

    /// Obtains an info set corresponding to the given key.
    let private getInfoSet infoSetKey infoSetMap numActions =
        match Map.tryFind infoSetKey infoSetMap with
            | Some infoSet ->
                assert(infoSet.RegretSum.Count = numActions)
                assert(infoSet.StrategySum.Count = numActions)
                infoSet
            | None ->
                InformationSet.zero numActions   // first visit

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

    /// Evaluates the utility of the given deal via counterfactual
    /// regret minimization.
    let private cfr infoSetMap playerCards communityCard =

        /// Top-level CFR loop.
        let rec loop (history : string) reachProbs =
            let rounds = history.Split('d')
            if LeducHoldem.isTerminal rounds then
                let payoff =
                    LeducHoldem.getPayoff
                        playerCards
                        communityCard
                        rounds
                float payoff, Seq.empty
            elif LeducHoldem.isRoundEnd (Array.last rounds) then
                let sign =
                    match history with
                        | "xbc" | "brc" -> -1.0
                        | _ -> 1.0
                let utility, keyedInfoSets =
                    loop (history + "d") reachProbs
                sign * utility, keyedInfoSets
            else
                let activePlayer =
                    (Array.last rounds).Length % LeducHoldem.numPlayers
                let infoSetKey =
                    if rounds.Length = 2 then
                        playerCards[activePlayer] + communityCard + history
                    else
                        playerCards[activePlayer] + history
                loopNonTerminal history activePlayer infoSetKey reachProbs

        /// Recurses for non-terminal game state.
        and loopNonTerminal history activePlayer infoSetKey reachProbs =

                // get info set for current state from this player's point of view
            let actions = LeducHoldem.getLegalActions history
            let infoSet = getInfoSet infoSetKey infoSetMap actions.Length

                // get player's current strategy for this info set
            let strategy = InformationSet.getStrategy infoSet

                // get utility of each action
            let actionUtilities, keyedInfoSets =
                let utilities, keyedInfoSetArrays =
                    (actions, strategy.ToArray())
                        ||> Array.map2 (fun action actionProb ->
                            let reachProbs =
                                updateReachProbabilities
                                    reachProbs
                                    activePlayer
                                    actionProb
                            loop (history + action) reachProbs)
                        |> Array.unzip
                getActiveUtilities utilities,
                Seq.concat keyedInfoSetArrays

                // utility of this info set is action utilities weighted by action probabilities
            let utility = actionUtilities * strategy

                // accumulate updated regrets and strategy
            let keyedInfoSets =
                let infoSet =
                    let regrets =
                        let opponent = (activePlayer + 1) % LeducHoldem.numPlayers
                        reachProbs[opponent] * (actionUtilities - utility)
                    let strategy =
                        reachProbs[activePlayer] * strategy
                    InformationSet.accumulate regrets strategy infoSet
                seq {
                    yield! keyedInfoSets
                    yield infoSetKey, infoSet
                }

            utility, keyedInfoSets

        [| 1.0; 1.0 |]
            |> DenseVector.ofArray
            |> loop ""

    /// Trains for the given number of iterations.
    let train numIterations =

            // all possible deals
        let permutations =
            List.permutations LeducHoldem.deck
                |> Seq.map (fun deck ->
                    Seq.toArray deck[0..1], deck[2])
                |> Seq.toArray

        let utilities, infoSetMap =

                // evaluate all permutations on each iteration
            let deals =
                seq {
                    for i = 1 to numIterations do
                        yield permutations[i % permutations.Length]
                }

                // start with no known info sets
            (Map.empty, deals)
                ||> Seq.mapFold (fun infoSetMap (playerCards, communityCards) ->

                        // evaluate one game starting with this deal
                    let utility, keyedInfoSets =
                        cfr infoSetMap playerCards communityCards

                        // update info sets
                    let infoSetMap =
                        (infoSetMap, keyedInfoSets)
                            ||> Seq.fold (fun acc (key, infoSet) ->
                                    Map.add key infoSet acc)

                    utility, infoSetMap)

            // compute average utility per deal
        let utility =
            Seq.sum utilities / float (permutations.Length * numIterations)
        utility, infoSetMap

let run () =

        // train
    let numIterations =
        if fsi.CommandLineArgs.Length > 1 then
            Int32.Parse(fsi.CommandLineArgs[1])
        else 10000
    printfn $"Running Leduc Hold'em vanilla CFR for {numIterations} iterations\n"
    let util, infoSetMap = LeducCfrTrainer.train numIterations

        // expected overall utility
    printfn $"Average game value for first player: %0.5f{util}\n"
    assert(abs(util - -1.0/18.0) <= 0.02)

        // strategy
    let strategyMap =
        infoSetMap
            |> Seq.map (fun (KeyValue(name, infoSet)) ->
                name, InformationSet.getAverageStrategy infoSet)
            |> Map
    let namedStrategies =
        strategyMap
            |> Map.toSeq
            |> Seq.sortBy (fst >> String.length)
    printfn "State   Bet     Check"
    for name, strategy in namedStrategies do
        let str =
            strategy
                |> Seq.map (sprintf "%0.5f")
                |> String.concat " "
        printfn $"%-3s{name}:    {str}"
    assert(
        let betAction = 0
        let k = strategyMap["K"][betAction]
        let j = strategyMap["J"][betAction]
        j >= 0.0 && j <= 1.0/3.0            // bet frequency for a Jack should be between 0 and 1/3
            && abs((k / j) - 3.0) <= 0.1)   // bet frequency for a King should be three times a Jack

let timer = Diagnostics.Stopwatch.StartNew()
run ()
printfn ""
printfn $"Elapsed time: {timer}"