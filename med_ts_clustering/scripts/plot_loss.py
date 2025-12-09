import matplotlib.pyplot as plt
import numpy as np

def save_loss_curve(loss_list, out_path):
    """
    loss_list: list of float, 每个 epoch 的 loss
    out_path: PNG 保存路径
    """

    plt.figure(figsize=(8, 5))

    x = np.arange(1, len(loss_list) + 1)

    # ----- 画曲线 -----
    plt.plot(x, loss_list, linewidth=2)

    # ----- 设置标题与字体 -----
    plt.title("MRM Training Loss Curve", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)

    # ----- 控制 x 轴刻度不太密集 -----
    if len(x) <= 10:
        plt.xticks(x)  # 少于等于10个 epoch直接全部显示
    else:
        step = max(1, len(x) // 10)   # 最多显示 10 个刻度
        ticks = np.arange(1, len(x) + 1, step)
        if ticks[-1] != len(x):       # 保证最后一个 epoch 一定显示
            ticks = np.append(ticks, len(x))
        plt.xticks(ticks)

    # ----- 网格（淡一点）-----
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
