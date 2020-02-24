"""
Import necessary packages
"""
from utils import *
import argparse
import multiprocessing as mp
from samplers import fastgcn_sampler, ladies_sampler, graphsage_sampler, exact_sampler, full_batch_sampler

"""
Dataset arguments
"""
parser = argparse.ArgumentParser(
    description='Training GCN on Large-scale Graph Datasets')

parser.add_argument('--dataset', type=str, default='pubmed',
                    help='Dataset name: pubmed/flickr/reddit/ppi')
parser.add_argument('--nhid', type=int, default=256,
                    help='Hidden state dimension')
parser.add_argument('--epoch_num', type=int, default=200,
                    help='Number of Epoch')
parser.add_argument('--pool_num', type=int, default=10,
                    help='Number of Pool')
parser.add_argument('--batch_num', type=int, default=10,
                    help='Maximum Batch Number')
parser.add_argument('--batch_size', type=int, default=512,
                    help='size of output node in a batch')
parser.add_argument('--n_layers', type=int, default=2,
                    help='Number of GCN layers')
parser.add_argument('--n_stops', type=int, default=200,
                    help='Early stops')
parser.add_argument('--samp_num', type=int, default=512,
                    help='Number of sampled nodes per layer (only for ladies & factgcn)')
parser.add_argument('--sample_method', type=str, default='graphsage',
                    help='Sampled Algorithms: ladies/fastgcn/graphsage/exact')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--sgd_lr', type=int, default=0.7,
                    help='learning rate for SGD')
args = parser.parse_args()
print(args)

"""
Prepare devices
"""
if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")

"""
Prepare data using multi-process
"""


def prepare_data(pool, sampler, process_ids, train_nodes, samp_num_list, num_nodes, lap_matrix, lap_matrix_sq, depth):
    jobs = []
    for _ in process_ids:
        batch_idx = torch.randperm(len(train_nodes))[:args.batch_size]
        batch_nodes = train_nodes[batch_idx]
        p = pool.apply_async(sampler, args=(np.random.randint(2**32 - 1), batch_nodes,
                                            samp_num_list, num_nodes, lap_matrix, lap_matrix_sq, depth))
        jobs.append(p)
    return jobs


lap_matrix, labels, feat_data, train_nodes, valid_nodes, test_nodes = preprocess_data(
    args.dataset, True if args.sample_method=='graphsage' else False)
print("Dataset information")
print(lap_matrix.shape, labels.shape, feat_data.shape,
      train_nodes.shape, valid_nodes.shape, test_nodes.shape)

if type(feat_data) == sp.lil.lil_matrix:
    feat_data = torch.FloatTensor(feat_data.todense()).to(device)
else:
    feat_data = torch.FloatTensor(feat_data).to(device)



"""
Setup datasets and models for training (multi-class use sigmoid+binary_cross_entropy, use softmax+nll_loss otherwise)
"""
if args.dataset in ['cora', 'citeseer', 'pubmed', 'flickr', 'reddit']:
    from model import GCN
    from optimizers import spider_step, multi_level_step_v1, multi_level_step_v2, sgd_step, full_step
    from optimizers import ForwardWrapper, package_mxl
    labels = torch.LongTensor(labels).to(device)
    num_classes = labels.max().item()+1
elif args.dataset in ['ppi', 'ppi-large', 'amazon', 'yelp']:
    from model_mc import GCN
    from optimizers_mc import spider_step, multi_level_step_v1, multi_level_step_v2, sgd_step, full_step
    from optimizers_mc import ForwardWrapper, package_mxl
    labels = torch.FloatTensor(labels).to(device)
    num_classes = labels.shape[1]

if args.sample_method == 'ladies':
    sampler = ladies_sampler
    args.samp_num = 512
    samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])
elif args.sample_method == 'fastgcn':
    args.samp_num = 512
    sampler = fastgcn_sampler
    samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])
elif args.sample_method == 'exact':
    sampler = exact_sampler
    samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)]) # never used
elif args.sample_method == 'graphsage':
    args.samp_num = 5
    sampler = graphsage_sampler
    samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])  # as proposed in GraphSage paper



def calculate_grad_variance(net, feat_data, labels, train_nodes, adjs_full):
    net_grads = []
    for p_net in net.parameters():
        net_grads.append(p_net.grad.data)
    clone_net = copy.deepcopy(net)
    _, _ = clone_net.calculate_loss_grad(
        feat_data, adjs_full, labels, train_nodes)

    clone_net_grad = []
    for p_net in clone_net.parameters():
        clone_net_grad.append(p_net.grad.data)
    del clone_net

    variance = 0.0
    for g1, g2 in zip(net_grads, clone_net_grad):
        variance += (g1-g2).norm(2) ** 2
    variance = torch.sqrt(variance)
    return variance



"""
This is a zeroth-order and first-order variance reduced version of Stochastic-GCN++
"""

def sgcn_pplus_01(feat_data, labels, lap_matrix, train_nodes, valid_nodes, test_nodes,  args, device, calculate_grad_vars=False):

    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)
    pool = mp.Pool(args.pool_num)
    lap_matrix_sq = lap_matrix.multiply(lap_matrix)
    jobs = prepare_data(pool, sampler, process_ids, train_nodes, samp_num_list, len(feat_data),
                        lap_matrix, lap_matrix_sq, args.n_layers)

    susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=num_classes,
                 layers=args.n_layers, dropout=args.dropout).to(device)
    susage.to(device)
    
    print(susage)

    adjs_full, input_nodes_full, sampled_nodes_full = full_batch_sampler(
        train_nodes, len(feat_data), lap_matrix, args.n_layers)
    adjs_full = package_mxl(adjs_full, device)

    forward_wrapper = ForwardWrapper(
        len(feat_data), args.nhid, args.n_layers, num_classes)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, susage.parameters()))

    best_model = copy.deepcopy(susage)
    best_val_loss, _ = susage.calculate_loss_grad(feat_data, adjs_full, labels, valid_nodes)
    cnt = 0
    
    loss_train = [best_val_loss]
    loss_test = [best_val_loss]
    grad_variance_all = []
    loss_train_all = [best_val_loss]

    for epoch in np.arange(args.epoch_num):
        # fetch train data
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()
        # prepare next epoch train data
        pool = mp.Pool(args.pool_num)
        jobs = prepare_data(pool, sampler, process_ids, train_nodes, samp_num_list, len(feat_data),
                            lap_matrix, lap_matrix_sq, args.n_layers)

        inner_loop_num = args.batch_num
        calculate_grad_vars = calculate_grad_vars and epoch<20
        cur_train_loss, cur_train_loss_all, grad_variance = multi_level_step_v2(susage, optimizer, feat_data, labels,
                                                                        train_nodes, valid_nodes, adjs_full, sampled_nodes_full,
                                                                        train_data, inner_loop_num, forward_wrapper, device, dist_bound=2e-3, #2e-4
                                                                        calculate_grad_vars=calculate_grad_vars)
        loss_train_all.extend(cur_train_loss_all)
        grad_variance_all.extend(grad_variance)
        # calculate validate loss
        susage.eval()

        susage.zero_grad()
        val_loss, _ = susage.calculate_loss_grad(
            feat_data, adjs_full, labels, valid_nodes)

        if val_loss < best_val_loss:
            best_model = copy.deepcopy(susage)
        if val_loss + 0.01 < best_val_loss:
            best_val_loss = val_loss
            cnt = 0
        else:
            cnt += 1
            
        if cnt == args.n_stops//args.batch_num:
            break

        cur_test_loss = val_loss

        loss_train.append(cur_train_loss)
        loss_test.append(cur_test_loss)
        
        # print progress
        print('Epoch: ', epoch,
              '| train loss: %.8f' % cur_train_loss,
              '| test loss: %.8f' % cur_test_loss)
    
    f1_score_test = best_model.calculate_f1(feat_data, adjs_full, labels, test_nodes)
    return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all


# In[8]:


"""
This is a first-order variance reduced version of Stochastic-GCN++
"""

def sgcn_pplus_1(feat_data, labels, lap_matrix, train_nodes, valid_nodes, test_nodes,  args, device, calculate_grad_vars=False):

    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)
    pool = mp.Pool(args.pool_num)
    lap_matrix_sq = lap_matrix.multiply(lap_matrix)
    jobs = prepare_data(pool, sampler, process_ids, train_nodes, samp_num_list, len(feat_data),
                        lap_matrix, lap_matrix_sq, args.n_layers)

    susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=num_classes,
                 layers=args.n_layers, dropout=args.dropout).to(device)
    susage.to(device)

    print(susage)

    adjs_full, input_nodes_full, sampled_nodes_full = full_batch_sampler(
        train_nodes, len(feat_data), lap_matrix, args.n_layers)
    adjs_full = package_mxl(adjs_full, device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, susage.parameters()))

    best_model = copy.deepcopy(susage)
    best_val_loss, _ = susage.calculate_loss_grad(feat_data, adjs_full, labels, valid_nodes)
    cnt = 0
    
    loss_train = [best_val_loss]
    loss_test = [best_val_loss]
    grad_variance_all = []
    loss_train_all = [best_val_loss]

    for epoch in np.arange(args.epoch_num):
        # fetch train data
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()
        # prepare next epoch train data
        pool = mp.Pool(args.pool_num)
        jobs = prepare_data(pool, sampler, process_ids, train_nodes, samp_num_list, len(feat_data),
                            lap_matrix, lap_matrix_sq, args.n_layers)

        inner_loop_num = args.batch_num
        calculate_grad_vars = calculate_grad_vars and epoch<20
        cur_train_loss, cur_train_loss_all, grad_variance = spider_step(susage, optimizer, feat_data, labels,
                                                         train_nodes, valid_nodes,
                                                         adjs_full, train_data, inner_loop_num, device,
                                                         calculate_grad_vars=calculate_grad_vars)
        loss_train_all.extend(cur_train_loss_all)
        grad_variance_all.extend(grad_variance)
        # calculate test loss
        susage.eval()

        susage.zero_grad()
        val_loss, _ = susage.calculate_loss_grad(
            feat_data, adjs_full, labels, valid_nodes)

        if val_loss < best_val_loss:
            best_model = copy.deepcopy(susage)
        if val_loss + 0.01 < best_val_loss:
            best_val_loss = val_loss
            cnt = 0
        else:
            cnt += 1
            
        if cnt == args.n_stops//args.batch_num:
            break

        cur_test_loss = val_loss

        loss_train.append(cur_train_loss)
        loss_test.append(cur_test_loss)
        
        # print progress
        print('Epoch: ', epoch,
              '| train loss: %.8f' % cur_train_loss,
              '| test loss: %.8f' % cur_test_loss)

    f1_score_test = best_model.calculate_f1(feat_data, adjs_full, labels, test_nodes)
    return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all



"""
This is a zeroth-order variance reduced version of Stochastic-GCN+
"""

def sgcn_pplus_0(feat_data, labels, lap_matrix, train_nodes, valid_nodes, test_nodes,  args, device, calculate_grad_vars=False):

    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)
    pool = mp.Pool(args.pool_num)
    lap_matrix_sq = lap_matrix.multiply(lap_matrix)
    jobs = prepare_data(pool, sampler, process_ids, train_nodes, samp_num_list, len(feat_data),
                        lap_matrix, lap_matrix_sq, args.n_layers)

    susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=num_classes,
                 layers=args.n_layers, dropout=args.dropout).to(device)
    susage.to(device)

    print(susage)

    adjs_full, input_nodes_full, sampled_nodes_full = full_batch_sampler(
        train_nodes, len(feat_data), lap_matrix, args.n_layers)
    adjs_full = package_mxl(adjs_full, device)

    # this stupid wrapper is only used for sgcn++
    forward_wrapper = ForwardWrapper(
        len(feat_data), args.nhid, args.n_layers, num_classes)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, susage.parameters()))

    best_model = copy.deepcopy(susage)
    best_val_loss, _ = susage.calculate_loss_grad(feat_data, adjs_full, labels, valid_nodes)
    cnt = 0
    
    loss_train = [best_val_loss]
    loss_test = [best_val_loss]
    grad_variance_all = []
    loss_train_all = [best_val_loss]

    for epoch in np.arange(args.epoch_num):
        # fetch train data
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()
        # prepare next epoch train data
        pool = mp.Pool(args.pool_num)
        jobs = prepare_data(pool, sampler, process_ids, train_nodes, samp_num_list, len(feat_data),
                            lap_matrix, lap_matrix_sq, args.n_layers)

        inner_loop_num = args.batch_num
        # compare with sgcn_plus, the only difference is we use multi_level_step_v1 here
        calculate_grad_vars = calculate_grad_vars and epoch<20
        cur_train_loss, cur_train_loss_all, grad_variance = multi_level_step_v1(susage, optimizer, feat_data, labels,
                                                                        train_nodes, valid_nodes, adjs_full, sampled_nodes_full,
                                                                        train_data, inner_loop_num, forward_wrapper, device, dist_bound=2e-3,
                                                                        calculate_grad_vars=calculate_grad_vars)
        loss_train_all.extend(cur_train_loss_all)
        grad_variance_all.extend(grad_variance)
        # calculate validate loss
        susage.eval()

        susage.zero_grad()
        val_loss, _ = susage.calculate_loss_grad(
            feat_data, adjs_full, labels, valid_nodes)

        if val_loss < best_val_loss:
            best_model = copy.deepcopy(susage)
        if val_loss + 0.01 < best_val_loss:
            best_val_loss = val_loss
            cnt = 0
        else:
            cnt += 1
            
        if cnt == args.n_stops//args.batch_num:
            break

        cur_test_loss = val_loss

        loss_train.append(cur_train_loss)
        loss_test.append(cur_test_loss)
        
        # print progress
        print('Epoch: ', epoch,
              '| train loss: %.8f' % cur_train_loss,
              '| test loss: %.8f' % cur_test_loss)

    f1_score_test = best_model.calculate_f1(feat_data, adjs_full, labels, test_nodes)
    return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all



"""
This is just an unchanged Stochastic-GCN 
"""


def sgcn(feat_data, labels, lap_matrix, train_nodes, valid_nodes, test_nodes,  args, device, calculate_grad_vars=False, full_batch=False):

    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)
    pool = mp.Pool(args.pool_num)
    lap_matrix_sq = lap_matrix.multiply(lap_matrix)
    jobs = prepare_data(pool, sampler, process_ids, train_nodes, samp_num_list, len(feat_data),
                        lap_matrix, lap_matrix_sq, args.n_layers)

    susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=num_classes,
                 layers=args.n_layers, dropout=args.dropout).to(device)
    susage.to(device)

    print(susage)

    adjs_full, input_nodes_full, sampled_nodes_full = full_batch_sampler(
        train_nodes, len(feat_data), lap_matrix, args.n_layers)
    adjs_full = package_mxl(adjs_full, device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, susage.parameters()))

    best_model = copy.deepcopy(susage)
    best_val_loss, _ = susage.calculate_loss_grad(feat_data, adjs_full, labels, valid_nodes)
    cnt = 0
    
    loss_train = [best_val_loss]
    loss_test = [best_val_loss]
    grad_variance_all = []
    loss_train_all = [best_val_loss]


    for epoch in np.arange(args.epoch_num):
        # fetch train data
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()
        # prepare next epoch train data
        pool = mp.Pool(args.pool_num)
        jobs = prepare_data(pool, sampler, process_ids, train_nodes, samp_num_list, len(feat_data),
                            lap_matrix, lap_matrix_sq, args.n_layers)

        inner_loop_num = args.batch_num

        # it can also run full-batch GD by ignoring all the samplings
        if full_batch:
            cur_train_loss, cur_train_loss_all, grad_variance = full_step(susage, optimizer, feat_data, labels,
                                    train_nodes, valid_nodes,
                                    adjs_full, train_data, inner_loop_num, device, 
                                    calculate_grad_vars=calculate_grad_vars)
        else:
            cur_train_loss, cur_train_loss_all, grad_variance = sgd_step(susage, optimizer, feat_data, labels,
                                              train_nodes, valid_nodes,
                                              adjs_full, train_data, inner_loop_num, device, 
                                              calculate_grad_vars=calculate_grad_vars)
        loss_train_all.extend(cur_train_loss_all)
        grad_variance_all.extend(grad_variance)
        # calculate test loss
        susage.eval()

        susage.zero_grad()
        val_loss, _ = susage.calculate_loss_grad(
            feat_data, adjs_full, labels, valid_nodes)

        if val_loss < best_val_loss:
            best_model = copy.deepcopy(susage)
        if val_loss + 0.01 < best_val_loss:
            best_val_loss = val_loss
            cnt = 0
        else:
            cnt += 1
            
        if cnt == args.n_stops//args.batch_num:
            break
            
        cur_test_loss = val_loss

        loss_train.append(cur_train_loss)
        loss_test.append(cur_test_loss)
        
        # print progress
        print('Epoch: ', epoch,
              '| train loss: %.8f' % cur_train_loss,
              '| test loss: %.8f' % cur_test_loss)
    f1_score_test = best_model.calculate_f1(feat_data, adjs_full, labels, test_nodes)
    return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all


fn = './results/{}_{}.pkl'.format(args.sample_method, args.dataset)
if not os.path.exists(fn):
    results = dict()
else:
    with open(fn, 'rb') as f:
        results = pkl.load(f)
calculate_grad_vars = True


st = time.time()
print('>>> sgcn')
susage, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all  = sgcn(
    feat_data, labels, lap_matrix, train_nodes, valid_nodes, test_nodes,  args, device, calculate_grad_vars, full_batch=False)
results['sgcn'] = [loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all]
print('sgcn', time.time() - st)


if args.sample_method is not 'exact':
    st = time.time()
    print('>>> sgcn_pplus_01')
    susage, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all = sgcn_pplus_01(
        feat_data, labels, lap_matrix, train_nodes, valid_nodes, test_nodes,  args, device, calculate_grad_vars)
    results['sgcn_pplus_01'] = [loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all]
    print('sgcn_pplus_01', time.time() - st)


if args.sample_method is not 'exact':
    st = time.time()
    print('>>> sgcn_pplus_0')
    susage, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all = sgcn_pplus_0(
        feat_data, labels, lap_matrix, train_nodes, valid_nodes, test_nodes,  args, device, calculate_grad_vars)
    results['sgcn_pplus_0'] = [loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all]
    print('sgcn_pplus_0', time.time() - st)


st = time.time()
print('>>> sgcn_pplus_1')
susage, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all = sgcn_pplus_1(
    feat_data, labels, lap_matrix, train_nodes, valid_nodes, test_nodes,  args, device, calculate_grad_vars)
results['sgcn_pplus_1'] = [loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all]
print('sgcn_pplus_1', time.time() - st)


results.keys()


with open(fn, 'wb') as f:
    pkl.dump(results, f)


