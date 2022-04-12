import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("./")


class ArgParserDataExtracter(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument('--figure_path', default=None, type=str,
                          help='A path for storing the images of the experiment.')
        self.add_argument('--figure_tag', default="", type=str,
                          help='Optional tag to be added to the name of the figure.')
        self.add_argument('--experiment_path', default=None, type=str,
                          help='A path to the directory containing the results of the experiments.')
        self.add_argument('--experiment_name', default=None,
                          help='The name of the set of experiments')
        self.add_argument('--interactions_cutoff', default=None, type=int,
                          help='Training data between this cutoff value is not taken into account')


class TrainingLogParser():
    def __init__(self, args):
        self.args = args
        self.total_reward = []
        self.file_list = self.find_all_files(self.args.experiment_path)
        if len(self.file_list) == 0:
            raise ValueError('No training logs found in this folder: {}'.format(self.args.experiment_path))

    def find_all_files(self, directory):
        file_list = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('log'):
                    file_list.append({"path": os.path.join(os.getcwd(), root, file), })
        return file_list

    def extract_data(self):
        for file in self.file_list:
            print("Extracting data from file: {}".format(file["path"]))
            test_reward, interactions = self.extract_logfile(file)
            file["test_reward"] = test_reward
            file["interactions"] = interactions
            self.total_reward.append(test_reward)
        return self.file_list

    def extract_logfile(self, file):
        test_reward = []
        step_count = 0
        f = open(file["path"], 'r')
        lines = f.read().splitlines()
        f.close()
        for line in lines:
            if not line:
                continue
            if line.startswith("Evaluation over 10 episodes:"):
                test_reward.append(float(line.split(",")[0].split(":")[-1]))
                step_count += 1

        interactions = np.arange(step_count) * 20000

        if self.args.interactions_cutoff is not None:
            remaining_index = np.argwhere(interactions <= self.args.interactions_cutoff)
            test_reward = np.array(test_reward)[remaining_index].flatten()
            interactions = interactions[remaining_index].flatten()

        return test_reward, interactions

    def compute_mean_std(self):
        reward_summary = np.array(self.total_reward)
        mean_reward = np.average(reward_summary, axis=0)
        std_reward = np.std(reward_summary, axis=0)
        return mean_reward, std_reward

    def generate_figs(self):
        steps = self.file_list[0]["interactions"]
        mean_reward, std_reward = self.compute_mean_std()
        max_reward = np.max(mean_reward)
        steps_till_converge = steps[np.argmax(mean_reward > (0.9 * max_reward))]
        print("Experiment name: {}, max_reward: {}, steps taken to converge: {}".format(self.args.experiment_name,
                                                                                        max_reward,
                                                                                        steps_till_converge))

        plt.figure(figsize=(8, 8))
        plt.plot(steps, mean_reward, label="Test Reward")
        plt.fill_between(steps, mean_reward + std_reward, mean_reward - std_reward, facecolor='blue', alpha=0.3)
        plt.title(self.args.experiment_name)
        plt.legend()
        plt.xlabel("Gradient Updates")
        plt.ylabel("Test Reward")
        if self.args.figure_path is not None:
            if not os.path.isdir(self.args.figure_path):
                os.makedirs(self.args.figure_path)
            fig_name = "{}_{}.png".format(self.args.experiment_name, self.args.figure_tag)
            plt.savefig(os.path.join(self.args.figure_path, fig_name))


if __name__ == "__main__":

    plt.style.use('science')
    plt.rcParams.update({'font.size': 11})

    parser = ArgParserDataExtracter()
    training_curve_list = []

    training_curve_list.append(
        {"experiment_path": "/home/tianxin/Desktop/experiment_discount", "label": "Strong-to-weak Curriculum"})
    training_curve_list.append({"experiment_path": "/home/tianxin/Desktop/experiment_fixed_power/no_buffer_data/100",
                                "label": "100\% " + r'$\mathcal{T}$'})
    training_curve_list.append({"experiment_path": "/home/tianxin/Desktop/experiment_fixed_power/no_buffer_data/70",
                                "label": "70\% " + r'$\mathcal{T}$'})
    training_curve_list.append({"experiment_path": "/home/tianxin/Desktop/experiment_fixed_power/no_buffer_data/60",
                                "label": "60\% " + r'$\mathcal{T}$'})
    training_curve_list.append({"experiment_path": "/home/tianxin/Desktop/experiment_fixed_power/no_buffer_data/50",
                                "label": "50\% " + r'$\mathcal{T}$'})
    training_curve_list.append({"experiment_path": "/home/tianxin/Desktop/experiment_fixed_power/no_buffer_data/40",
                                "label": "40\% " + r'$\mathcal{T}$'})
    # training_curve_list.append({"experiment_path": "experiment/rsi_comparison/eps_rsi","label":r'$\epsilon$'+'-RSI'})
    # training_curve_list.append({"experiment_path": "experiment/rsi_comparison/rsi","label":"RSI"})
    # training_curve_list.append({"experiment_path": "experiment/rsi_comparison/no_rsi","label":"Without RSI"})

    # training_curve_list.append({"experiment_path": "/home/tianxin/Desktop/experiment_discount","label":"Torque Curriculum"})

    for training_curve in training_curve_list:
        args = parser.parse_args()
        args.experiment_path = training_curve["experiment_path"]
        trainingLogParser = TrainingLogParser(args)
        file_list = trainingLogParser.extract_data()
        training_curve["mean"], training_curve["variance"] = trainingLogParser.compute_mean_std()
        training_curve["interactions"] = file_list[0]["interactions"]

    # plt.figure(figsize=(11, 9))
    for (i, training_curve) in enumerate(training_curve_list):
        plt.plot(training_curve["interactions"] / 1000000, training_curve["mean"], color="C{}".format(i),
                 label=training_curve["label"])
        plt.fill_between(training_curve["interactions"] / 1000000, \
                         (training_curve["mean"] + training_curve["variance"]),
                         (training_curve["mean"] - training_curve["variance"]), \
                         color="C{}".format(i), alpha=0.1)

    plt.legend()
    # plt.title("HumanoidStandup")
    plt.xlabel("Environment Steps " + r'($\times 10^6$)')
    plt.ylabel("Average Reward")
    # plt.ylabel("Torque Limit (\% of " + r'$\mathcal{T}$' + ")")
    plt.savefig("./figs/reward_curriculum.png", dpi=300)
    plt.show()
