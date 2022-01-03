import Exercise3.game.tictactoe as t
from tqdm import tqdm

p1, p2 = "X", "O"

for amt in [1000, 10000, 100000, 1000000]:
    results = []
    for i in tqdm(range(amt)):
        game = t.Game(p1, p2)
        game.sim_random(False)
        results.append(game.get_result())

    counts = ["draw:" + str(results.count(t.Player.neutral)), "p1:" + str(results.count(p1)),
              "p2:" + str(results.count(p2))]
    percentages = ["draw:" + str(results.count(t.Player.neutral) / amt), "p1:" + str(results.count(p1) / amt),
                   "p2:" + str(results.count(p2) / amt)]

    with open("testcounts.txt", "a") as file:
        file.write("\n" + str(amt) + "\n" + "  ".join(counts) + "\n" + "  ".join(percentages) + "\n")
        file.close()
