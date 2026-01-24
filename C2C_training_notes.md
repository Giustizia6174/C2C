## C2C 项目：training recipe / 新模型组合 / evaluation 学习笔记

### 一、任务理解：看 C2C 的 training recipe、理解训练逻辑

在这个仓库里，训练流程采用「**配置文件（recipe）+ 通用训练脚本**」的形式：

- `recipe/train_recipe/*.json`：描述 **要训什么**、**怎么训**：
  - 用哪对模型：`base_model`（接收者）+ `teacher_model`（分享者）
  - projector 类型和超参：`projector.type`, `params`
  - 训练超参：学习率、batch size、epoch、梯度累积、scheduler 等
  - 数据集类型和大小：`data.type`, `kwargs`
  - 输出路径、保存频率、wandb 信息
- `script/train/SFT_train.py`：统一的 **训练入口脚本**，读取 recipe，把模型、数据、优化器、DDP 等都搭好，然后跑一个通用的 training loop，只更新 projector 参数。
- `rosetta/train/*`：封装数据集适配、collator、mapping 工具函数等。

「看和学习 training recipe，理解训练逻辑」的含义是：
**读懂一个典型 recipe（如 `C2C_0.6+0.5.json`），再沿着它在 `SFT_train.py` 里如何被使用，搞清楚从配置到实际训练步骤的对应关系。**

---

### 二、如何系统地阅读 training recipe 和训练逻辑

#### 1. 从 README 入手，定位入口

README 里已经给出推荐入口：

- **训练配置文件位置**：`recipe/train_recipe/`（如 `C2C_0.6+0.5.json`）
- **训练脚本**：`script/train/SFT_train.py`
- **示例运行命令**：

```bash
# 单 GPU
python script/train/SFT_train.py --config recipe/train_recipe/C2C_0.6+0.5.json

# 多 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=8 script/train/SFT_train.py \
  --config recipe/train_recipe/C2C_0.6+0.5.json
```

#### 2. 精读一个典型 recipe：`recipe/train_recipe/C2C_0.6+0.5.json`

可以重点关注几个部分（只列结构，不抄完整内容）：

- **`model` 部分**：
  - `base_model`: `"Qwen/Qwen3-0.6B"`
  - `teacher_model`: `"Qwen/Qwen2.5-0.5B-Instruct"`
  - `projector`：`type: "C2CProjector"` + 一系列超参（hidden_dim, num_layers, dropout, temperature 等）
  - `mapping`: `"last_aligned"`（说明 base/teacher 层之间如何对齐）
- **`training` 部分**：
  - 学习率、epoch、`max_length`
  - `per_device_train_batch_size`, `gradient_accumulation_steps`
  - `freeze`: `["teacher","base"]`（表示只训练 projector）
- **`output` 部分**：
  - `output_dir`: 训练 checkpoint 的输出目录（最终会有 `final/` 子目录）
- **`data` 部分**：
  - `type`: 数据集类型（对应 `rosetta/train/dataset_adapters.py` 中的配置）
  - `kwargs`: 具体 split、样本数、最大长度等
  - `train_ratio`: train / eval 划分比例

阅读时可以带着这样的问题：

- 这份配置想训练哪一对模型？
- projector 的结构和超参是什么样的？
- 训练到底只更新哪些参数？（`freeze` 字段）
- 数据从哪里来，如何被切分成 train / eval？

#### 3. 再看训练脚本 `script/train/SFT_train.py`

推荐阅读路径：

1. **配置加载与拆分**

   - `main()` 中通过 `--config` 读取 JSON/YAML：
     - `cfg = load_config(args.config)`
     - 拆成 `model_config`, `training_config`, `output_config`, `data_config`
   - 这一步建立起「配置文件字段 → Python 变量 → 后续用法」的映射。

2. **区分 baseline / rosetta 训练模式**

   - `detect_training_mode(model_config)`：
     - 有 `baseline_model` 且没有 `base_model` → baseline 单模型训练
     - 有 `base_model` 和 `teacher_model` → rosetta（C2C）训练
   - 我们主要关注 rosetta 路径。

3. **模型构建逻辑：`setup_models(model_config, training_mode, ...)`**

   rosetta 分支的大致流程：

   - 用 `AutoTokenizer.from_pretrained` 加载 base 模型 tokenizer；
   - 用 `AutoModelForCausalLM.from_pretrained` 加载 base / teacher 模型；
   - 自动计算 base / teacher 的 hidden_dim / num_heads / num_layers；
   - 读取 `model_config["projector"]`，通过 `create_projector` 按层构造 `projector_list`；
   - 初始化 `RosettaModel(model_list=[base_model, teacher_model], projector_list=projector_list, ...)`；
   - 根据 `mapping` 字段选择 layer mapping 策略：
     - `last_aligned_sources(...)`
     - 或 `k_nearest_sources(...)`
   - 对每个 target layer 调用 `set_projector_config` 绑定：
     - 哪个 teacher layer 的 KV-cache
     - 到哪个 base layer
     - 使用哪一个 projector index
   - 如果 `is_do_alignment = true`，还会构造 `TokenAligner` 做 token-level 对齐。

4. **参数冻结逻辑**

- 读取 `training_config["freeze"]`（如 `["teacher","base"]`）：
  - `"base"` 在 `model.model_list[0]` 上调用 `freeze_model`
  - `"teacher"` 在 `model.model_list[1]` 上调用 `freeze_model`
  - projector 列表通过 `unfreeze_projectors(model)` 确保是可训练的
- 这就实现了「只训 projector」的典型 C2C 设定。

5. **数据与 collator：`rosetta/train/dataset_adapters.py`**

- 用 `create_dataset(dataset_type=data_config["type"], **data_config["kwargs"])` 构造原始数据集；
- 按模式（baseline / rosetta）选择：
  - `BaselineChatDataset`
  - 或 `ChatDataset` / `AlignedChatDataset`
- 再用：
  - `BaselineDataCollator`
  - 或 `RosettaDataCollator` 组 batch；
- 对 rosetta 来说，collator 还会构造：
  - `kv_cache_index`
  - `position_ids`
  - 以及 labels 等训练所需张量。

6. **训练循环：`train_step` 和外层 loop**

- `train_step()` 中 rosetta 分支：

  - 从 batch 取出 `input_ids`, `attention_mask`, `position_ids`, `kv_cache_index`, `labels`；
  - 调用：

    ```python
    outputs = model.forward(
      kv_cache_index=kv_cache_index,
      input_ids=input_ids,
      attention_mask=attention_mask,
      position_ids=position_ids,
      labels=labels,
      use_cache=True
    )
    loss = outputs.loss
    ```

  - 梯度只会回到 projector（因为 base / teacher 已被冻结）。

- 外层训练循环负责：
  - DDP / gradient accumulation
  - scheduler 更新
  - 中间 evaluation
  - checkpoint 保存（`projector_i.pt` + `projector_i.json` + `projector_config.json` 等）

**总结的阅读顺序（Checklist）：**

1. `README` 中「Train C2C Projectors」小节（整体流程）
2. `recipe/train_recipe/C2C_0.6+0.5.json`（具体实验配置）
3. `script/train/SFT_train.py`：
   - `main()` 配置加载与拆分
   - `setup_models()` rosetta 路径
   - `freeze` / `unfreeze_projectors` 逻辑
   - `train_step()` rosetta 分支
   - checkpoint 保存部分
4. `rosetta/train/dataset_adapters.py`（只看与你用到的 `data.type` 相关的类和函数）

---

### 三、训练新的模型组合：如 Qwen3-4B + Qwen2.5-3B（matching size pair）

#### 1. 任务含义

仓库中已经提供了一些支持的模型组合（README 里列出了 Qwen3-0.6B + Qwen2.5-0.5B、Qwen3-0.6B + Qwen3-4B 等）。

现在要做的是：

- 在这个框架里新加一对模型，比如：
  - base（receiver）：`Qwen/Qwen3-4B-Base`（名字以实际 HF 模型为准）
  - teacher（sharer）：`Qwen/Qwen2.5-3B-Instruct`
- 用 C2C 思路训练 projector，让 4B 模型能够利用 3B 模型的 KV-cache，实现语义融合。

本质上：**不是训练一个新 LLM，而是在两只现成 LLM 之间训练 KV-cache projector。**

#### 2. 操作步骤 / 配置指导

1. **复制已有 recipe 作为模板**

   - 复制 `recipe/train_recipe/C2C_0.6+0.5.json` → `C2C_4b+3b.json`
   - 修改 `model` 段：

     - `base_model`: `"Qwen/Qwen3-4B-Base"`
     - `teacher_model`: `"Qwen/Qwen2.5-3B-Instruct"`
     - `projector` 的 `params` 可以先沿用 0.6+0.5 的设置
     - `mapping`: 先用 `"last_aligned"`，之后再尝试 `"k_nearest"` 做 ablation

   - 修改 `training` 段：
     - 4B + 3B 组合显存占用会比 0.6+0.5 大很多，需要：
       - 降低 `per_device_train_batch_size`
       - 或降低 `max_length`
       - 或多卡（`torchrun`）训练

   - 修改 `output` 段：

     - 比如：

       ```json
       "output_dir": "local/checkpoints/qwen3_4b+qwen2.5_3b_C2C"
       ```

     - 最终会产生 `local/checkpoints/qwen3_4b+qwen2.5_3b_C2C/final` 目录，供推理和评测使用。

2. **确认 HuggingFace 模型可用**

   - 确认 `Qwen/Qwen3-4B-Base`、`Qwen/Qwen2.5-3B-Instruct` 能在当前环境用 transformers 正常 `from_pretrained`；
   - 可以先写一个小脚本简单加载并生成一句话测试。

3. **运行训练**

   ```bash
   # 单卡试通（估算显存）
   CUDA_VISIBLE_DEVICES=0 python script/train/SFT_train.py \
     --config recipe/train_recipe/C2C_4b+3b.json

   # 多卡训练（示例）
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   torchrun --nproc_per_node=4 script/train/SFT_train.py \
     --config recipe/train_recipe/C2C_4b+3b.json
   ```

4. **关于 “matching size pair” 的考虑**

- C2C 的 projector 内部已经能处理 hidden_dim / head 数不一致，所以框架层面并不强制相同规模；
- 「matching size」更多是实验设计上的考量（让 base/teacher 容量相近、能力更匹配）；
- 在 recipe 里主要需要关注：
  - 模型能加载且显存压力可控；
  - batch size / max_length 合理；
  - mapping / projector 结构可以先沿用已有成功配置，再在此基础上做实验调整。

---

### 四、Evaluate：用已有评测代码评估训练好的 C2C 模型

#### 1. 任务含义

训练出 projector 后，会有一个 `.../final` checkpoint 目录，需要：

- 用这个 checkpoint 重新组装 RosettaModel；
- 在标准 benchmark（MMLU-Redux、MMMLU、LongBench、GSM8K 等）上评测；
- 查看整体准确率以及各科目 / 子类 / 大类的表现。

仓库已经提供了完整的统一评测框架：

- **配置文件**：`recipe/eval_recipe/unified_eval.yaml`
- **评测脚本**：`script/evaluation/unified_evaluator.py`
- **底层工具**：`rosetta/utils/evaluate.py`

#### 2. 评测配置：`recipe/eval_recipe/unified_eval.yaml`

核心结构（节选）：

```yaml
model:
  model_name: Rosetta
  rosetta_config:
    base_model: Qwen/Qwen3-0.6B
    teacher_model: Qwen/Qwen2.5-0.5B-Instruct
    checkpoints_dir: local/checkpoints/0.6+0.5B_C2C_general_again/final

  generation_config:
    do_sample: false
    max_new_tokens: 64

output:
  output_dir: local/final_results/0.6+0.5B_C2C_general_again

eval:
  dataset: mmlu-redux
  gpu_ids: [0]
  answer_method: generate
  use_cot: false
  use_template: true
```

对你自己的新组合，需要修改：

- `rosetta_config.base_model` / `teacher_model`；
- `rosetta_config.checkpoints_dir`：指向刚训练好的 `.../final`；
- `output.output_dir`：改为新的结果路径名。

#### 3. 统一评测脚本：`script/evaluation/unified_evaluator.py`

阅读要点：

- `main()`：
  - 读取 `--config` 对应 YAML；
  - 构造 `UnifiedEvaluator(config)`，然后调用 `run()`。

- `UnifiedEvaluator.__init__`：
  - 解析 `model` / `output` / `eval`；
  - 根据 `eval.dataset` 选择对应的 `DATASET_CONFIGS`；
  - 保存 `generation_config`、`gpu_ids`、`answer_method` 等设置。

- **模型加载逻辑**（`evaluate_on_gpu`）：

  - 如果 `model_name` 中包含 `"rosetta"`：
    - 调用 `load_rosetta_model(self.model_config, self.eval_config, device, generation_config)`；
    - 该函数定义在 `rosetta/utils/evaluate.py`：
      - 根据 `rosetta_config.base_model` / `teacher_model` 加载底层模型；
      - 根据 `checkpoints_dir` 加载 `projector_*.pt/json` 等；
      - 设置 `projector_dict` 等配置；
      - 返回 `(rosetta_model, tokenizer)`。

- **单个 subject 的评测**（`evaluate_subject`）：

  - 从 HuggingFace 或本地加载对应数据集；
  - 使用 `_format_*_example` 系列函数构造 prompt；
  - 按 `answer_method`：
    - `logits`：只看最后一个 token 上 A/B/C/D 的 logits；
    - `generate`：完整生成后用 `extract_answer_from_content` 抽取答案；
  - 统计 correctness、长度信息、CoT 日志。

- 多 GPU / 多 subject 分发在 `run()`、`merge_results()`、`save_results()` 中实现，理解即可。

#### 4. 最低使用步骤

以现成的 0.6+0.5 为例：

1. 在 `unified_eval.yaml` 中确认 `checkpoints_dir` 指向训练好的 `.../final`。
2. 运行：

   ```bash
   python script/evaluation/unified_evaluator.py \
     --config recipe/eval_recipe/unified_eval.yaml
   ```

3. 结果输出：

   - summary JSON：整体 accuracy、各 subject、各 category/subcategory；
   - length 统计 JSON：输入长度、生成长度、长度比；
   - CoT / answer CSV：记录题目、选项、答案、预测、输出文本等。

对你自己的 4B+3B 组合，只需复制一份 `unified_eval.yaml`（比如命名 `unified_eval_4b+3b.yaml`），改掉模型和 checkpoint 路径，按同样方式运行即可。

---

### 五、三步行动总结

1. **理解训练逻辑**  
   - 按顺序阅读 README → `C2C_0.6+0.5.json` → `SFT_train.py` → `dataset_adapters.py`，建立起「配置 → 模型结构 → projector 训练 → 数据管线」的全局图。

2. **设计并训练新的 Qwen3-4B + Qwen2.5-3B 组合**  
   - 复制已有 recipe，改模型名与输出目录；
   - 按显存调整 `per_device_train_batch_size` 和 `max_length`；
   - 用 `SFT_train.py` 跑通训练并拿到 `final/` checkpoint。

3. **使用统一评测框架进行 evaluate**  
   - 复制 `unified_eval.yaml` 并改为你的新模型配置与 checkpoint 路径；
   - 用 `unified_evaluator.py` 在 MMLU-Redux / MMMLU / 其它任务上跑评测，分析结果并与原始组合对比。


