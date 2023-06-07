# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tf_geometric as tfg
from tf_geometric.datasets.ppi import PPIDataset
from tf_geometric.utils.graph_utils import RandomNeighborSampler
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

#加载数据集
train_graphs, valid_graphs, test_graphs = PPIDataset().load_data()

# 遍历所有图
for graph in train_graphs + valid_graphs + test_graphs:
    neighbor_sampler = RandomNeighborSampler(graph.edge_index)
    graph.cache["sampler"] = neighbor_sampler

num_classes = train_graphs[0].y.shape[1]

#包含两个 MaxPoolingGraphSage 层的列表。每个 MaxPoolingGraphSage 层都有一个激活函数为 relu，
#设置输出单元数为 128。这些层用于从输入特征中提取图节点的表征。
graph_sages = [
    # tfg.layers.MaxPoolGraphSage(units=256, activation=tf.nn.relu, concat=True),
    # tfg.layers.MaxPoolGraphSage(units=256, activation=tf.nn.relu, concat=True)

    # tfg.layers.MeanPoolGraphSage(units=256, activation=tf.nn.relu, concat=True),
    # tfg.layers.MeanPoolGraphSage(units=256, activation=tf.nn.relu, concat=True)

    tfg.layers.MeanGraphSage(units=256, activation=tf.nn.relu, concat=True),
    tfg.layers.MeanGraphSage(units=256, activation=tf.nn.relu, concat=True)

    # tfg.layers.SumGraphSage(units=256, activation=tf.nn.relu, concat=True),
    # tfg.layers.SumGraphSage(units=256, activation=tf.nn.relu, concat=True)

    # tfg.layers.LSTMGraphSage(units=256, activation=tf.nn.relu, concat=True),
    # tfg.layers.LSTMGraphSage(units=256, activation=tf.nn.relu, concat=True)

    # tfg.layers.GCNGraphSage(units=256, activation=tf.nn.relu),
    # tfg.layers.GCNGraphSage(units=256, activation=tf.nn.relu)
]
# fc为包含一个 Dropout 层和一个全连接层的序列模型。Dropout 层用于在训练过程中进行随机失活，有助于防止过拟合。
#全连接层用于将最后一层的节点表征映射到预定义的类别数。
fc = tf.keras.Sequential([
    keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes)
])
#指定每个 GraphSAGE 层采样邻居节点的数量。
num_sampled_neighbors_list = [25, 10]

#接收图对象和训练标志（用于控制是否应用 Dropout），返回预测结果。
def forward(graph, training=False):
    neighbor_sampler = graph.cache["sampler"]
    h = graph.x
    for i, (graph_sage, num_sampled_neighbors) in enumerate(zip(graph_sages, num_sampled_neighbors_list)):
        sampled_edge_index, sampled_edge_weight = neighbor_sampler.sample(k=num_sampled_neighbors)
        h = graph_sage([h, sampled_edge_index, sampled_edge_weight], training=training)
    h = fc(h, training=training)
    return h

#损失函数
def compute_loss(logits, vars):
    #计算交叉熵损失，将 logits 和标签之间的交叉熵作为损失值。
    losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits,
        labels=tf.convert_to_tensor(graph.y, dtype=tf.float32)
    )
    #从变量列表中筛选出的包含 "kernel" 的变量，对应模型中的权重参数。
    kernel_vars = [var for var in vars if "kernel" in var.name]
    #每个权重参数的 L2 正则化项的损失。
    l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vars]
    #返回交叉熵损失和 L2 正则化项之和作为最终的损失值。
    return tf.reduce_mean(losses) + tf.add_n(l2_losses) * 1e-5


def calc_f1(y_true, y_pred):
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = 0

    return f1_score(y_true, y_pred, average="micro")

#评估模型
def evaluate(graphs):
    #存储预测结果
    y_preds = []
    #存储真实标签
    y_true = []
    #遍历图数据中的每个对象，将真实标签和预测标签分别加到对应的数组中
    for graph in graphs:
        y_true.append(graph.y)
        logits = forward(graph)
        y_preds.append(logits.numpy())
    y_pred = np.concatenate(y_preds, axis=0)
    y = np.concatenate(y_true, axis=0)
    #计算F1的得分数
    mic = calc_f1(y, y_pred)
    #返回计算得到的F1分数，F1分数作为模型在给定图数据集上的评估结果。
    return mic

#根据梯度更新模型参数，学习率设置为0.01
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

for epoch in tqdm(range(80)):
    for graph in train_graphs:
        # 创建梯度带，记录模型参数的梯度信息。
        with tf.GradientTape() as tape:
            logits = forward(graph, training=True)
            #计算损失函数
            loss = compute_loss(logits, tape.watched_variables())
        # 获取梯度带中的变量列表，即模型参数
        vars = tape.watched_variables()
        grads = tape.gradient(loss, vars)
        #更新模型参数
        optimizer.apply_gradients(zip(grads, vars))
    #检查是否满足打印条件
    if epoch % 1 == 0:
        #评估模型在测试集上的性能，计算F1
        valid_f1_mic = evaluate(valid_graphs)
        test_f1_mic = evaluate(test_graphs)
        print("epoch = {}\tloss = {}\tvalid_f1_micro = {}".format(epoch, loss, valid_f1_mic))
        print("epoch = {}\ttest_f1_micro = {}".format(epoch, test_f1_mic))
# test_f1_mic = evaluate(test_graphs)
# print("test_f1_micro = {}".format(test_f1_mic))