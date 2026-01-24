"""
仅用于从 HuggingFace Hub 下载 C2C 的 projector/aggregator 等 checkpoint 文件，
不会加载基础模型，也不会执行推理。

下载完成后，会打印可直接作为 `checkpoints_dir` 使用的路径。
"""

import os

from huggingface_hub import snapshot_download


def download_c2c_checkpoint(
    repo_id: str = "nics-efc/C2C_Fuser",
    subfolder: str = "qwen3_0.6b+qwen3_4b_Fuser",
) -> str:
    """
    从 HuggingFace Hub 下载 C2C 预训练 checkpoint。

    Args:
        repo_id: 远端仓库名。
        subfolder: 仓库中具体使用的子目录名称。

    Returns:
        该子目录下 `final` 目录的绝对路径，作为 checkpoints_dir 使用。
    """
    checkpoint_root = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{subfolder}/*"],
    )

    # 组合出 README 示例中使用的 final 目录
    checkpoints_dir = os.path.join(checkpoint_root, subfolder, "final")
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(
            f"Expected checkpoints_dir='{checkpoints_dir}' does not exist. "
            "Please check the repo structure or subfolder name."
        )
    return checkpoints_dir


def main() -> None:
    """
    主函数：只负责下载 C2C checkpoint 并打印可用的 checkpoints_dir 路径。

    不加载任何基础模型，不执行推理。
    """
    print("==> 下载/准备 C2C projector/aggregator checkpoint ...")
    checkpoints_dir = download_c2c_checkpoint()
    print(f"checkpoints_dir: {checkpoints_dir}")
    print("✅ 下载完成。请在 inference 脚本中将 `checkpoints_dir` 设置为以上路径。")


if __name__ == "__main__":
    main()



