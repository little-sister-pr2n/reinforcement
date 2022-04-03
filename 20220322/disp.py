import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def main():
    results = os.listdir("./results")
    # fig, axs = plt.subplots(1,2, figsize=(8,4))

    # axs_idx = 0
    # handles = list()
    # labels = list()
    for env in results:
        env_dir = os.listdir(f"./results/{env}")
        # print(env)
        
        for algo in env_dir:
            algo_dir = os.listdir(f"./results/{env}/{algo}")
            # print(f"\t{algo}")
            mean_algo_per_seed = np.zeros((20,5))
            stdev_algo_per_seed = np.zeros((20,5))
            idx = 0
            for seed in algo_dir:
                seed_dir = os.listdir(f"./results/{env}/{algo}/{seed}")
                seed_dir2 = os.listdir(f"./results/{env}/{algo}/{seed}/{seed_dir[0]}")
                # print(f"\t\t{seed}")
                score = pd.read_table(f"./results/{env}/{algo}/{seed}/{seed_dir[0]}/{seed_dir2[-2]}")
                mean_algo_per_seed[:,idx] = score["mean"]
                stdev_algo_per_seed[:,idx] = score["stdev"]
                idx += 1
            mean_algo_data = mean_algo_per_seed.mean(1)
            stdev_algo_data = stdev_algo_per_seed.mean(1)
            steps = score["steps"]
            # print(stdev_algo_data)
            # print(mean_algo_data)
            plt.plot(steps, mean_algo_data, label=algo)
            plt.fill_between(steps, mean_algo_data-stdev_algo_data, mean_algo_data+stdev_algo_data, alpha=0.3)
        plt.xlabel("steps")
        plt.ylabel("cumulative rewards")
        plt.title(env)
        plt.legend()
        plt.savefig(f"./figure/{env}.pdf")
        plt.clf()
        plt.close()
if __name__ == "__main__":
    main()