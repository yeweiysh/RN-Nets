from itertools import product

import argparse
from datasets import get_dataset
from train_eval import cross_validation_with_val_set

from gcn import GCN, GCNWithJK
from graph_sage import GraphSAGE, GraphSAGEWithJK
from gin import GIN0, GIN0WithJK, GIN, GINWithJK
from graclus import Graclus
from top_k import TopK
from sag_pool import SAGPool
from diff_pool import DiffPool
from edge_pool import EdgePool
from global_attention import GlobalAttentionNet
from set2set import Set2SetNet
from sort_pool import SortPool
from asap import ASAP
from kgcn import KGCN
from rnnets import RNNETS

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
args = parser.parse_args()

hiddens = [32,64]

datasets = ['IMDB-BINARY','IMDB-MULTI','PTC_FM','PTC_MM','PTC_FR','PTC_MR','MUTAG','ENZYMES','NCI109','NCI1','PROTEINS','REDDIT-BINARY','REDDIT-MULTI-5K','COLLAB','DD','KKI', 'BZR_MD', 'COX2_MD', 'DHFR','BZR', 'COX2']

nets = [
    # GCNWithJK,
    # GraphSAGEWithJK,
    # GIN0WithJK,
    # GINWithJK,
    # Graclus,
    # TopK,
    # SAGPool,
    # DiffPool,
    # EdgePool,
    # GCN,
    # GraphSAGE,
    # GIN0,
    # GIN,
    # GlobalAttentionNet,
    # Set2SetNet,
    # SortPool,
    # ASAP,
    RNNETS
]

def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    train_loss = info['train_loss']
    test_acc = info['test_acc']
    #print('{:02d}/{:03d}: Train Loss: {:.3f}: Test Accuracy: {:.3f}'.format(
    #    fold, epoch, train_loss, test_acc))


results = []
for dataset_name, Net in product(datasets, nets):
    best_result = (0, 0)  # (acc, std)
    print('-----\n{} - {}'.format(dataset_name, Net.__name__))
    for  hidden in hiddens:
        dataset = get_dataset(dataset_name, sparse=Net != DiffPool)
        model = Net(dataset, hidden)
        acc, std = cross_validation_with_val_set(
            dataset,
            model,
            folds=10,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            logger=logger,
        )
        if acc > best_result[0]:
            best_result = (acc, std)

    desc = '{:.3f} ± {:.3f}'.format(best_result[0], best_result[1])
    print('Best result - {}'.format(desc))
    results += ['{} - {}: {}'.format(dataset_name, model, desc)]
print('-----\n{}'.format('\n'.join(results)))
