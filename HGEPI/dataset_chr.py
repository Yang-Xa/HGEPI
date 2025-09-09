import torch
from torch_geometric.utils import negative_sampling
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T


def data_choice(name, split_chr):
    EPG_data = HeteroData()
    enhancer = torch.load(f'/home/yx/rasha/people_gene/new_data/{name}/{name}_enhancer_features.pth')
    promoter = torch.load(f'/home/yx/rasha/people_gene/new_data/{name}/{name}_promoter_features.pth')
    pos_edges_dict = torch.load(f'/home/yx/rasha/people_gene/new_data/{name}/{name}_edges_dict.pth')
    neg_edges_dict = torch.load(f'/home/yx/rasha/people_gene/new_data/{name}/{name}_negative_edges_dict.pth')

    test_chr = (split_chr % 10) + 1

    val_edges = pos_edges_dict[split_chr]
    neg_val_edges = neg_edges_dict[split_chr]

    test_edges = pos_edges_dict[test_chr]
    neg_test_edges = neg_edges_dict[test_chr]
    
    train_edges_list = []
    neg_train_edges_list = []
    
    for chr_num in pos_edges_dict:
        if chr_num != split_chr and chr_num != test_chr:
            train_edges_list.append(pos_edges_dict[chr_num])
            neg_train_edges_list.append(neg_edges_dict[chr_num])
    
    train_edges = torch.cat(train_edges_list, dim=1)
    neg_train_edges = torch.cat(neg_train_edges_list, dim=1)
    
    EPG_data['promoter'].x = promoter
    EPG_data['enhancer'].x = enhancer
    EPG_data['promoter', 'interactive', 'enhancer'].edge_index = train_edges
    EPG_data = T.ToUndirected()(EPG_data)
    
    return EPG_data, promoter, enhancer, train_edges, val_edges, test_edges, neg_train_edges, neg_val_edges, neg_test_edges


class edge_Dataset(torch.utils.data.Dataset):
    def __init__(self, pos, neg):
        self.edge = torch.cat([pos, neg], dim=1)
        pos = torch.ones(pos.size(1), dtype=torch.int64)
        neg = torch.zeros(neg.size(1), dtype=torch.int64)
        self.labels = torch.cat([pos, neg], dim=0)

    def __getitem__(self, index):
        return [self.edge.T[index][0], self.edge.T[index][1]], self.labels[index]

    def __len__(self):
        return self.edge.size(1)


def create_loaders(name='GM12878', split_chr=1, batch_size=64):
    result = data_choice(name, split_chr)
    EPG_data, promoter, enhancer = result[0], result[1], result[2]
    train_edges, val_edges, test_edges = result[3], result[4], result[5]
    neg_train_edges, neg_val_edges, neg_test_edges = result[6], result[7], result[8]
    
    trainset = edge_Dataset(train_edges, neg_train_edges)
    valset = edge_Dataset(val_edges, neg_val_edges)
    testset = edge_Dataset(test_edges, neg_test_edges)
    
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset=valset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=batch_size,
        shuffle=True
    )
    
    return train_loader, val_loader, test_loader, EPG_data, promoter, enhancer


