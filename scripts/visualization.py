import matplotlib.pyplot as plt
import numpy as np

from dm_control.utils import rewards

plt.style.use('science')
plt.rcParams['figure.dpi'] = 300


# plt.rc('legend',fontsize=5)

def generate_reward_diagram():
    x = np.linspace(-2, 2, num=200)
    y = rewards.tolerance(x, bounds=(-0.5, 0.5), margin=1)

    fig, ax = plt.subplots()
    fig.set_size_inches(3.3, 1.8)

    ax.plot(x, y)

    ax.axvline(x=-0.5, ymax=0.95, linestyle="--", color='C2')
    ax.axvline(x=0.5, ymax=0.95, linestyle="--", color='C2')
    ax.axvline(x=-1.5, ymax=0.12, linestyle="--", color='C2')
    ax.axvline(x=1.5, ymax=0.12, linestyle="--", color='C2')

    ax.axhline(y=0.1, xmax=0.161, linestyle="--", color='C2')
    ax.axhline(y=1.0, xmax=0.375, linestyle="--", color='C2')

    x_positions = (-1.5, -0.5, 0.5, 1.5)
    x_labels = (r'$b_{l}-m$', r'$b_{l}$', r'$b_{u}$', r'$b_{u}+m$')
    plt.xticks(x_positions, x_labels)

    y_positions = (0, 0.1, 0.25, 0.5, 0.75, 1.0)
    y_labels = ("0", "v", "0.25", "0.5", "0.75", "1.0")
    plt.yticks(y_positions, y_labels)

    plt.xlabel("Input i")
    plt.ylabel("Reward Function f")

    plt.tight_layout()

    plt.savefig("./figs/reward.png", dpi=600)

    plt.show()


def generate_velocity_trajectory():
    trajectory_list = []
    symbol = ["-o", "-*", "-."]

    # controller1
    # trajectory_list.append({
    #     "path": "final_results/final_controllers/HumanoidStandup_01-06-01-13_seed_0_no_move_run_1/results/slow/RecordedMotionSlow1.npz",
    #     "label": "Slow Speed",
    #     "marker_size": 2
    # })
    # trajectory_list.append({
    #     "path": "final_results/final_controllers/HumanoidStandup_01-06-01-13_seed_0_no_move_run_1/results/fast/RecordedMotionSlow1.npz",
    #     "label": "Fast Speed",
    #     "marker_size": 2
    # })
    # trajectory_list.append({
    #     "path": "final_results/final_controllers/HumanoidStandup_01-06-01-13_seed_0_no_move_run_1/results/reference/RecordedMotionFast1.npz",
    #     "label": "Reference",
    #     "marker_size": 4
    # })

    # controller2
    # trajectory_list.append({
    #     "path": "final_results/final_controllers/HumanoidStandup_01-07-20-29_seed_0_no_move_run_8/results/slow/RecordedMotionSlow1.npz",
    #     "label": "Slow Speed",
    #     "marker_size": 2
    # })
    # trajectory_list.append({
    #     "path": "final_results/final_controllers/HumanoidStandup_01-07-20-29_seed_0_no_move_run_8/results/fast/RecordedMotionSlow1.npz",
    #     "label": "Fast Speed",
    #     "marker_size": 2
    # })
    # trajectory_list.append({
    #     "path": "final_results/final_controllers/HumanoidStandup_01-07-20-29_seed_0_no_move_run_8/results/reference/RecordedMotionFast1.npz",
    #     "label": "Reference",
    #     "marker_size": 4
    # })

    # # controller3
    # trajectory_list.append({
    #     "path": "final_results/final_controllers/HumanoidStandup_01-11-06-52_seed_0_no_move_new_run_8/results/slow/RecordedMotionSlow1.npz",
    #     "label": "Slow Speed",
    #     "marker_size": 2
    # })
    # trajectory_list.append({
    #     "path": "final_results/final_controllers/HumanoidStandup_01-11-06-52_seed_0_no_move_new_run_8/results/fast/RecordedMotionSlow1.npz",
    #     "label": "Fast Speed",
    #     "marker_size": 2
    # })
    # trajectory_list.append({
    #     "path": "final_results/final_controllers/HumanoidStandup_01-11-06-52_seed_0_no_move_new_run_8/results/reference/RecordedMotionFast1.npz",
    #     "label": "Reference",
    #     "marker_size": 4
    # })

    # controller5
    trajectory_list.append({
        "path": "final_results/final_controllers/HumanoidStandup_01-14-01-34_seed_0_no_move_run_0/results/slow/RecordedMotionSlow1.npz",
        "label": "Slow Speed",
        "marker_size": 2
    })
    trajectory_list.append({
        "path": "final_results/final_controllers/HumanoidStandup_01-14-01-34_seed_0_no_move_run_0/results/fast/RecordedMotionSlow1.npz",
        "label": "Fast Speed",
        "marker_size": 2
    })
    trajectory_list.append({
        "path": "final_results/final_controllers/HumanoidStandup_01-14-01-34_seed_0_no_move_run_0/results/reference/RecordedMotionFast1.npz",
        "label": "Reference",
        "marker_size": 4
    })

    for i, trajectory in enumerate(trajectory_list):
        data = np.load(trajectory["path"])
        rho = np.sqrt(data["head_pos"][:, 0] ** 2 + data["head_pos"][:, 1] ** 2)
        height = data["head_pos"][:, 2]
        plt.plot(rho, height, symbol[i], label=trajectory["label"], color="C{}".format(i),
                 markersize=trajectory["marker_size"])

    plt.legend()
    plt.xlabel("Distance to Origin")
    plt.ylabel("Head Height")
    plt.savefig("./figs/head_trajectory_4.png", dpi=300)

    plt.show()


def generate_rate_diagram():
    for k in [0.75, 0.5, 0.25]:
        # assume the length of the reference motion is 50
        y = np.arange(50)
        x = y / k
        # plt.figure(figsize=(3.3, 0.9), dpi=200)
        plt.plot(x, y, label=r"$\kappa={}$".format(k))

    k = 0.5
    y = np.arange(50)
    x = np.arange(0, 160, 2)
    for _ in range(30):
        y = np.insert(y, 20, y[20])
    plt.plot(x, y, label="Pauses")

    plt.xlim([0, 200])
    plt.ylim([0, 50])
    # plt.xticks([])
    # plt.yticks([])
    plt.legend()
    plt.xlabel("Input Frame Index")
    plt.ylabel("Reference Frame Index")
    plt.savefig("./figs/different_rates.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    generate_rate_diagram()
