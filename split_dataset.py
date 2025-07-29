import pandas as pd
import numpy as np

def update_domain_masks(domain_csv, ood_csv, test_ratio=0.1, random_state=42, id_col='domain', save_path=None):
    domain = pd.read_csv(domain_csv)
    ood = pd.read_csv(ood_csv)
    ood_malicious_mask = domain[id_col].isin(ood[id_col])
    domain['ood_malicious_mask'] = ood_malicious_mask
    domain.loc[ood_malicious_mask, 'label'] = 1
    # 1. 找到所有ood malicious的索引
    ood_indices = domain[domain[id_col].isin(ood[id_col])].index
    print(len(ood_indices))
    n_ood = len(ood_indices)
    n_ood_test = int(n_ood * test_ratio)
    np.random.seed(random_state)
    ood_test_indices = np.random.choice(ood_indices, n_ood_test, replace=False)
    ood_remain_indices = list(set(ood_indices) - set(ood_test_indices))

    # 2. test_mask = 测试集ood malicious + 随机等量label=0
    label0_indices = domain[domain['label'] == 0].index
    label0_for_test = np.random.choice(label0_indices, n_ood_test, replace=False)
    test_mask = np.zeros(len(domain), dtype=bool)
    test_mask[ood_test_indices] = True
    test_mask[label0_for_test] = True

    # 3. val_mask = 等量label=1 + 等量label=0
    label1_indices = domain[(domain['label'] == 1) & ~ood_malicious_mask].index  # 排除ood malicious
    label1_for_val = np.random.choice(label1_indices, n_ood_test, replace=False)
    label0_for_val = np.random.choice(list(set(label0_indices) - set(label0_for_test)), n_ood_test, replace=False)
    val_mask = np.zeros(len(domain), dtype=bool)
    val_mask[label1_for_val] = True
    val_mask[label0_for_val] = True

    # 4. train_mask_zs: 非ood的label=1和label=0，且不在test/val
    non_ood_indices = domain[~domain[id_col].isin(ood[id_col])].index
    train_mask_zs = np.zeros(len(domain), dtype=bool)
    for idx in non_ood_indices:
        if not (test_mask[idx] or val_mask[idx]):
            train_mask_zs[idx] = domain.loc[idx, 'label'] in [0, 1]

    # 5. train_mask_pre: 所有ood malicious + label=1 + label=0，且不在test/val
    train_mask_pre = np.zeros(len(domain), dtype=bool)
    for idx in range(len(domain)):
        if not (test_mask[idx] or val_mask[idx]):
            if (idx in ood_indices and not test_mask[idx]):
                train_mask_pre[idx] = True
            elif domain.loc[idx, 'label'] in [0, 1]:
                train_mask_pre[idx] = True

    # 6. train_mask_finetune: 所有ood malicious + 同数量label=1 + 上述两者数量之和的label=0（都不在test/val）
    # 先选ood_remain
    n_ood_remain = len(ood_remain_indices)
    # 再选同数量label=1
    label1_for_finetune = np.random.choice(list(set(label1_indices) - set(label1_for_val)), n_ood_remain, replace=False)
    # 再选(ood_remain+label1_for_finetune)数量的label=0
    label0_for_finetune = np.random.choice(
        list(set(label0_indices) - set(label0_for_test) - set(label0_for_val)),
        n_ood_remain * 2, replace=False
    )
    train_mask_finetune = np.zeros(len(domain), dtype=bool)
    for idx in ood_remain_indices:
        if not (test_mask[idx] or val_mask[idx]):
            train_mask_finetune[idx] = True
    for idx in label1_for_finetune:
        if not (test_mask[idx] or val_mask[idx]):
            train_mask_finetune[idx] = True
    for idx in label0_for_finetune:
        if not (test_mask[idx] or val_mask[idx]):
            train_mask_finetune[idx] = True

    # 7. 写入domain
    domain['test_mask'] = test_mask
    domain['val_mask'] = val_mask
    domain['train_mask_zs'] = train_mask_zs
    domain['train_mask_pre'] = train_mask_pre
    domain['train_mask_finetune'] = train_mask_finetune
    ood_in_test = domain[domain['val_mask'] & domain['ood_malicious_mask']].shape[0]
    print(f"val_mask中有 {ood_in_test} 个ood_malicious_mask为True")
    # 8. 保存
    if save_path is None:
        save_path = domain_csv
    domain.to_csv(save_path, index=False)
    print(f"Updated domain saved to {save_path}")


def process_iochg_nodes(node_csv, save_path=None):
    """
    处理iochg benchmark中的url和file文件
    
    Args:
        node_csv: url或file的CSV文件路径
        save_path: 保存路径，如果为None则覆盖原文件
    """
    # 读取数据
    nodes = pd.read_csv(node_csv)
    print(len(nodes))
    # 1. 将原本label=1的替换为0，label=0的替换为1
    nodes['label1'] = nodes['label']  # 保存原始label
    nodes.loc[nodes['label1'] == 1, 'label'] = 0
    nodes.loc[nodes['label1'] == 0, 'label'] = 1
    
    # 2. 将label<2的train_mask设置为true
    train_mask = nodes['label'] < 2
    nodes['train_mask'] = train_mask
    
    # 3. 将label>=2的一半设置为val_mask为true，一半test_mask为true
    label_ge_2_indices = nodes[nodes['label'] >= 2].index
    n_label_ge_2 = len(label_ge_2_indices)
    n_half = n_label_ge_2 // 2
    
    # 随机打乱索引
    np.random.seed(42)
    shuffled_indices = np.random.permutation(label_ge_2_indices)
    
    # 前一半设为val_mask，后一半设为test_mask
    val_indices = shuffled_indices[:n_half]
    test_indices = shuffled_indices[n_half:]
    
    # 初始化mask
    val_mask = np.zeros(len(nodes), dtype=bool)
    test_mask = np.zeros(len(nodes), dtype=bool)
    
    # 设置mask
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    nodes['val_mask'] = val_mask
    nodes['test_mask'] = test_mask
    
    # 打印统计信息
    print(f"处理文件: {node_csv}")
    print(f"总样本数: {len(nodes)}")
    print(f"label < 2 (train_mask): {train_mask.sum()}")
    print(f"label >= 2 中设为val_mask: {val_mask.sum()}")
    print(f"label >= 2 中设为test_mask: {test_mask.sum()}")
    print(f"原始label=0变为label=1: {(nodes['label1'] == 0).sum()}")
    print(f"原始label=1变为label=0: {(nodes['label1'] == 1).sum()}")
    print(len(nodes))
    # 保存
    if save_path is None:
        save_path = node_csv
    nodes.to_csv(save_path, index=False)
    print(f"Updated file saved to {save_path}")
    
    return nodes



def process_iochg_small_test(domain_csv, ood_csv, id_col='content', save_path=None):
    """
    处理iochg_small，仅用于测试场景
    
    Args:
        domain_csv: domain的CSV文件路径
        ood_csv: ood malicious的CSV文件路径
        id_col: 用于匹配的列名
        save_path: 保存路径，如果为None则覆盖原文件
    """
    # 读取数据
    domain = pd.read_csv(domain_csv)
    ood = pd.read_csv(ood_csv)
    
    # 1. 设置ood_malicious_mask
    ood_malicious_mask = domain[id_col].isin(ood[id_col])
    domain['ood_malicious_mask'] = ood_malicious_mask
    domain.loc[ood_malicious_mask, 'label'] = 1
    
    # 2. 找到所有ood malicious的索引
    ood_indices = domain[domain[id_col].isin(ood[id_col])].index
    n_ood = len(ood_indices)
    print(f"ood malicious数量: {n_ood}")
    
    # 3. test_mask = 所有ood malicious + 同数量的label=0 domain
    label0_indices = domain[domain['label'] == 0].index
    label0_for_test = np.random.choice(label0_indices, n_ood, replace=False)
    
    test_mask = np.zeros(len(domain), dtype=bool)
    test_mask[ood_indices] = True  # 所有ood malicious
    test_mask[label0_for_test] = True  # 同数量的label=0
    
    # 4. val_mask = 同数量的label=1 domain + 同数量的label=0 domain
    # 从剩余的label=1中选择
    label1_indices = domain[(domain['label'] == 1) & ~ood_malicious_mask].index  # 排除ood malicious
    label1_for_val = np.random.choice(label1_indices, n_ood, replace=False)
    
    # 从剩余的label=0中选择
    remaining_label0 = list(set(label0_indices) - set(label0_for_test))
    label0_for_val = np.random.choice(remaining_label0, n_ood, replace=False)
    
    val_mask = np.zeros(len(domain), dtype=bool)
    val_mask[label1_for_val] = True
    val_mask[label0_for_val] = True
    
    # 5. 写入domain
    domain['test_mask'] = test_mask
    domain['val_mask'] = val_mask
    # 不设置train_mask，因为仅用于测试
    
    # 6. 打印统计信息
    ood_in_test = domain[domain['test_mask'] & domain['ood_malicious_mask']].shape[0]
    ood_in_val = domain[domain['val_mask'] & domain['ood_malicious_mask']].shape[0]
    print(f"test_mask中有 {ood_in_test} 个ood_malicious_mask为True")
    print(f"val_mask中有 {ood_in_val} 个ood_malicious_mask为True")
    print(f"test_mask总样本数: {test_mask.sum()}")
    print(f"val_mask总样本数: {val_mask.sum()}")
    
    # 7. 保存
    if save_path is None:
        save_path = domain_csv
    domain.to_csv(save_path, index=False)
    print(f"Updated domain saved to {save_path}")
    
    return domain

# 使用示例
# process_iochg_small_test(
#     '/data1/hongjiegu/dataset/IOCHeteroGraph/dataset/IOCHeteroGraph/Data-2023-1212-1537-zcf/nodes/domain_new',
#     '/home/hongjiegu/projects/GRAVEL/dataset/iochg-dgl/ood_malicious_1537.csv',
#     id_col='content',
#     save_path='/data1/hongjiegu/dataset/IOCHeteroGraph/dataset/IOCHeteroGraph/Data-2023-1212-1537-zcf/nodes/domain_new'
# )

# 使用示例
# # 处理url文件
process_iochg_nodes(
    '/data1/hongjiegu/dataset/IOCHeteroGraph/dataset/IOCHeteroGraph/Data-2022-1114-1605-a/nodes/url_nodes',
    save_path='/data1/hongjiegu/dataset/IOCHeteroGraph/dataset/IOCHeteroGraph/Data-2022-1114-1605-a/nodes/url_nodes'
)

# 处理file文件
process_iochg_nodes(
    '/data1/hongjiegu/dataset/IOCHeteroGraph/dataset/IOCHeteroGraph/Data-2022-1114-1605-a/nodes/file_nodes',
    save_path='/data1/hongjiegu/dataset/IOCHeteroGraph/dataset/IOCHeteroGraph/Data-2022-1114-1605-a/nodes/file_nodes'
)

# 处理iochg_small的url和file文件
process_iochg_nodes(
    '/data1/hongjiegu/dataset/IOCHeteroGraph/dataset/IOCHeteroGraph/Data-2023-1212-1537-zcf/nodes/url_nodes',
    save_path='/data1/hongjiegu/dataset/IOCHeteroGraph/dataset/IOCHeteroGraph/Data-2023-1212-1537-zcf/nodes/url_nodes'
)

process_iochg_nodes(
    '/data1/hongjiegu/dataset/IOCHeteroGraph/dataset/IOCHeteroGraph/Data-2023-1212-1537-zcf/nodes/file_nodes',
    save_path='/data1/hongjiegu/dataset/IOCHeteroGraph/dataset/IOCHeteroGraph/Data-2023-1212-1537-zcf/nodes/file_nodes'
)

# update_domain_masks(
#     '/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/domain.csv',
#     '/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/ood_pdns.csv',
#     test_ratio=0.1,
#     id_col='domain',
#     save_path='/home/hongjiegu/projects/GRAVEL/dataset/pdns-dgl/domain.csv'
# )


# update_domain_masks(
#     '/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/domain.csv',
#     '/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/ood_minta.csv',
#     test_ratio=0.1,
#     id_col='value',
#     save_path='/home/hongjiegu/projects/GRAVEL/dataset/minta-dgl/domain.csv'
# )
update_domain_masks(
    '/data1/hongjiegu/dataset/IOCHeteroGraph/dataset/IOCHeteroGraph/Data-2022-1114-1605-a/nodes/domain_new',
    '/home/hongjiegu/projects/GRAVEL/dataset/iochg-dgl/ood_malicious_2022_1114_1605.csv',
    test_ratio=0.1,
    id_col='content',
    save_path='/data1/hongjiegu/dataset/IOCHeteroGraph/dataset/IOCHeteroGraph/Data-2022-1114-1605-a/nodes/domain_new'
)


