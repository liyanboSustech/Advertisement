# HSTU模型实现

基于Meta的生成式推荐系统中的HSTU（Hierarchical Sequential Transduction Unit）架构的PyTorch实现。

## 文件结构

### 核心文件
- **`dataset.py`** - 数据集处理和特征转换
  - `MyDataset`: 数据集类，支持多种特征类型
  - 特征处理：稀疏特征、数组特征、连续特征、嵌入特征
  - 支持多worker数据加载

- **`model.py`** - HSTU模型实现
  - `HSTUModel`: 主要的HSTU模型类
  - 支持InfoNCE损失函数
  - 支持多种特征类型处理
  - 序列编码和预测功能
  - 候选库嵌入生成

- **`main.py`** - 训练和推理主程序
  - 完整的训练流程
  - 模型验证和保存
  - 物品嵌入生成
  - TensorBoard日志记录

### 辅助文件
- **`hstu_components.py`** - HSTU核心组件
  - 嵌入模块、输入预处理器、输出后处理器

- **`simplified_hstu.py`** - 简化的HSTU实现
  - HSTU注意力机制
  - 序列编码器

- **`dot_product_similarity.py`** - 点积相似度模块
  - 为HSTU提供相似度计算

## 主要特性

### 1. HSTU架构
- **分层序列转换单元**: 专门为序列推荐设计的注意力机制
- **SiLU激活**: 使用Sigmoid Linear Unit激活函数
- **因果掩码**: 确保模型只能看到历史信息
- **相对位置编码**: 结合时间和位置信息

### 2. 特征处理
- **多种特征类型**: 支持稀疏、数组、连续、嵌入特征
- **特征预处理**: 自动处理不同类型的特征
- **批处理**: 高效的批处理特征转换
- **多worker兼容**: 支持多进程数据加载

### 3. 损失函数
- **InfoNCE损失**: 信息噪声对比估计损失
- **温度缩放**: 可调节的温度参数
- **归一化**: 训练和推理时的一致归一化

### 4. 训练优化
- **AdamW优化器**: 带权重衰减的Adam优化器
- **余弦退火学习率**: 自动调整学习率
- **梯度裁剪**: 防止梯度爆炸
- **早停机制**: 基于验证损失的早停

## 使用方法

### 1. 训练模型

```bash
python main.py \
    --data_dir ./data \
    --output_dir ./outputs \
    --log_dir ./logs \
    --batch_size 128 \
    --lr 0.001 \
    --hidden_units 64 \
    --num_blocks 2 \
    --num_heads 1 \
    --attention_dim 64 \
    --linear_dim 64 \
    --dropout_rate 0.2 \
    --num_epochs 10 \
    --infonce_temp 0.1 \
    --device cuda \
    --num_workers 4
```

### 2. 推理和生成嵌入

```bash
python main.py \
    --inference_only \
    --state_dict_path ./outputs/model_epoch_5.pt \
    --data_dir ./data \
    --output_dir ./outputs \
    --device cuda
```

### 3. 主要参数说明

#### HSTU模型参数
- `--hidden_units`: 隐藏层维度 (默认: 64)
- `--num_blocks`: HSTU块数量 (默认: 2)
- `--num_heads`: 注意力头数 (默认: 1)
- `--attention_dim`: 注意力维度 (默认: 64)
- `--linear_dim`: 线性层维度 (默认: 64)
- `--dropout_rate`: Dropout率 (默认: 0.2)

#### 训练参数
- `--batch_size`: 批次大小 (默认: 128)
- `--lr`: 学习率 (默认: 0.001)
- `--num_epochs`: 训练轮数 (默认: 3)
- `--weight_decay`: 权重衰减 (默认: 0.01)
- `--infonce_temp`: InfoNCE损失温度 (默认: 0.1)

#### 数据参数
- `--maxlen`: 序列最大长度 (默认: 101)
- `--mm_emb_id`: 多模态特征ID (默认: ['81'])
- `--num_workers`: 数据加载工作进程数 (默认: 4)

## 数据格式要求

### 输入数据文件
- `train_data.bin`: 训练数据
- `train_offsets.bin`: 训练数据偏移量
- `valid_data.bin`: 验证数据
- `valid_offsets.bin`: 验证数据偏移量
- `test_data.bin`: 测试数据
- `test_offsets.bin`: 测试数据偏移量

### 特征文件
- `item_feat_dict.json`: 物品特征字典
- `feature_types.json`: 特征类型定义
- `feat_statistics.json`: 特征统计信息
- `feature_default_value.json`: 特征默认值
- `indexer.pkl`: 索引映射

### 多模态特征
- `creative_emb/`: 多模态嵌入目录
  - `81.pkl`, `82.pkl`, ...: 不同模态的嵌入文件

## 模型架构

```
HSTUModel
├── embedding_module (LocalEmbeddingModule)
├── input_preprocessor (LearnablePositionalEmbeddingInputFeaturesPreprocessor)
├── hstu (HSTU)
│   ├── blocks (HSTUAttention × num_blocks)
│   │   ├── qkv_proj
│   │   ├── output_proj
│   │   ├── layer_norm
│   │   └── dropout
│   └── _causal_mask
├── output_postprocessor (L2NormEmbeddingPostprocessor)
└── sparse_emb / emb_transform (额外特征处理)
```

## 性能优化

### 1. 内存优化
- 使用梯度检查点
- 混合精度训练
- 特征预处理批处理

### 2. 速度优化
- 多worker数据加载
- PIN内存使用
- 异步数据加载

### 3. 训练稳定性
- 梯度裁剪
- 学习率预热
- 权重初始化优化

## 输出结果

### 模型文件
- `model_epoch_X.pt`: 训练过程中的模型检查点

### 嵌入文件
- `item_embeddings/embedding.fbin`: 物品嵌入向量
- `item_embeddings/id.u64bin`: 物品ID映射

### 日志文件
- TensorBoard日志: 训练损失、学习率等指标

## 扩展功能

### 1. 自定义特征类型
在`dataset.py`中添加新的特征处理逻辑

### 2. 修改损失函数
在`model.py`中修改`compute_infonce_loss`方法

### 3. 添加新的注意力机制
在`simplified_hstu.py`中扩展`HSTUAttention`类

### 4. 集成其他评估指标
在`main.py`中添加验证和测试逻辑

## 依赖项

```
torch>=1.9.0
numpy>=1.19.0
tqdm>=4.60.0
tensorboard>=2.4.0
```

## 注意事项

1. **数据格式**: 确保输入数据格式正确
2. **内存使用**: 大规模数据集可能需要调整批次大小
3. **GPU内存**: HSTU模型可能需要较多GPU内存
4. **特征类型**: 确保特征类型定义与实际数据匹配

## 参考论文

- "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"
  - https://arxiv.org/abs/2402.17152