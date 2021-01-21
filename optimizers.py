from utils import *


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
    return variance.cpu()


def package_mxl(mxl, device):
    return [torch.sparse.FloatTensor(mx[0], mx[1], mx[2]).to(device) for mx in mxl]


"""
SGCN++ (first-order only variance reduction) wrapper
"""

def sgcn_first(net, optimizer, feat_data, labels,
                train_nodes, valid_nodes,
                adjs_full, input_nodes_full, output_nodes_full, sampled_nodes_full,
                train_data, inner_loop_num, device, dist_bound=5e-3, calculate_grad_vars=False):
    """
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    net.train()

    transfer_time = 0
    compute_time = 0  

    # record previous net full gradient
    pre_net_full = copy.deepcopy(net)
    # record previous net mini batch gradient
    pre_net_mini = copy.deepcopy(net)

    # Compute full grad
    compute_time_start = time.perf_counter()
    optimizer.zero_grad()
    current_loss, _ = net.calculate_loss_grad(
        feat_data[input_nodes_full], adjs_full, labels[output_nodes_full], np.arange(len(output_nodes_full)))
    compute_time = time.perf_counter() - compute_time_start
    optimizer.step()

    running_loss = [current_loss.cpu().detach()]
    iter_num = 0

    grad_variance = []
    # Run over the train_loader
    while iter_num < inner_loop_num:
        for adjs, input_nodes, output_nodes, sampled_nodes in train_data:
            transfer_time_start = time.perf_counter()
            adjs = package_mxl(adjs, device)
            transfer_time = transfer_time + time.perf_counter() - transfer_time_start

            # record previous net full gradient
            for p_net, p_full in zip(net.parameters(), pre_net_full.parameters()):
                p_full.grad = copy.deepcopy(p_net.grad)

            # compute previous stochastic gradient
            pre_net_mini.zero_grad()
            # take backward
            compute_time_start = time.perf_counter()
            pre_net_mini.partial_grad(
                feat_data[input_nodes], adjs, labels[output_nodes])
            compute_time = compute_time + time.perf_counter() - compute_time_start

            # compute current stochastic gradient
            optimizer.zero_grad()

            compute_time_start = time.perf_counter()
            current_loss = net.partial_grad(
                feat_data[input_nodes], adjs, labels[output_nodes])
            compute_time = compute_time + time.perf_counter() - compute_time_start

            # take SCSG gradient step
            for p_net, p_mini, p_full in zip(net.parameters(), pre_net_mini.parameters(), pre_net_full.parameters()):
                p_net.grad.data = p_net.grad.data + p_full.grad.data - p_mini.grad.data

            # only for experiment purpose to demonstrate ...
            if calculate_grad_vars:
                grad_variance.append(calculate_grad_variance(
                    net, feat_data, labels, train_nodes, adjs_full))

            # record previous net mini batch gradient
            for p_mini, p_net in zip(pre_net_mini.parameters(), net.parameters()):
                p_mini.data = copy.deepcopy(p_net.data)

            optimizer.step()

            # print statistics
            running_loss += [current_loss.cpu().detach()]
            iter_num += 1.0

    # calculate training loss
    train_loss = np.mean(running_loss)

    return train_loss, running_loss, grad_variance, {'compute_time': compute_time, 'transfer_time':transfer_time}


"""
SGD wrapper
"""
def sgd_step(net, optimizer, feat_data, labels,
             train_nodes, valid_nodes,
             adjs_full, train_data, inner_loop_num, device, calculate_grad_vars=False):
    """
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    net.train()

    compute_time = 0
    transfer_time = 0

    running_loss = []
    iter_num = 0.0

    grad_variance = []
    # Run over the train_loader
    while iter_num < inner_loop_num:
        for adjs, input_nodes, output_nodes, sampled_nodes in train_data:

            transfer_time_start = time.perf_counter()
            adjs = package_mxl(adjs, device)
            transfer_time = transfer_time + time.perf_counter() - transfer_time_start

            # compute current stochastic gradient
            optimizer.zero_grad()
            compute_time_start = time.perf_counter()
            current_loss = net.partial_grad(
                feat_data[input_nodes], adjs, labels[output_nodes])
            compute_time = compute_time + time.perf_counter() - compute_time_start

            # only for experiment purpose to demonstrate ...
            if calculate_grad_vars:
                grad_variance.append(calculate_grad_variance(
                    net, feat_data, labels, train_nodes, adjs_full))

            optimizer.step()

            # print statistics
            running_loss += [current_loss.cpu().detach()]
            iter_num += 1.0

    # calculate training loss
    train_loss = np.mean(running_loss)

    return train_loss, running_loss, grad_variance, {'compute_time': compute_time, 'transfer_time':transfer_time}


"""
Full-batch
"""
def full_step(net, optimizer, feat_data, labels,
              train_nodes, valid_nodes,
              adjs_full, inner_loop_num, device, calculate_grad_vars=False):
    """
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    net.train()

    compute_time = 0
    transfer_time = 0

    running_loss = []
    iter_num = 0.0

    grad_variance = []
    # Run over the train_loader
    while iter_num < inner_loop_num:

        # compute current stochastic gradient
        optimizer.zero_grad()
        compute_time_start = time.perf_counter()
        current_loss, _ = net.calculate_loss_grad(
            feat_data, adjs_full, labels, train_nodes)
        compute_time = compute_time + time.perf_counter() - compute_time_start
        
        # only for experiment purpose to demonstrate ...
        if calculate_grad_vars:
            grad_variance.append(calculate_grad_variance(
                net, feat_data, labels, train_nodes, adjs_full))
        optimizer.step()

        # print statistics
        running_loss += [current_loss.cpu().detach()]
        iter_num += 1.0

    # calculate training loss
    train_loss = np.mean(running_loss)

    return train_loss, running_loss, grad_variance, {'compute_time': compute_time, 'transfer_time':transfer_time}


"""
Used for SGCN+ (Zeroth-order variance reduction) and SGCN++ (Doubly variance reduction)
"""


class ForwardWrapper(nn.Module):
    def __init__(self, n_nodes, n_hid, n_layers, n_classes):
        super(ForwardWrapper, self).__init__()
        self.n_layers = n_layers
        self.hiddens = torch.zeros(n_layers, n_nodes, n_hid)

    def forward_mini(self, net, staled_net, x, adjs, sampled_nodes):
        transfer_time = 0
        compute_time = 0

        cached_outputs = []
        for ell in range(self.n_layers):
            transfer_time_start = time.perf_counter()
            stale_x = x if ell == 0 else self.hiddens[ell - 1, sampled_nodes[ell-1]].to(x)
            transfer_time = transfer_time + time.perf_counter()-transfer_time_start

            compute_time_start = time.perf_counter()
            stale_x = staled_net.gcs[ell](stale_x, adjs[ell])
            stale_x = staled_net.dropout(staled_net.relu(stale_x))
            x = net.gcs[ell](x, adjs[ell])
            x = net.dropout(net.relu(x))
            compute_time = compute_time + time.perf_counter()-compute_time_start

            transfer_time_start = time.perf_counter()
            x = x + self.hiddens[ell, sampled_nodes[ell]].to(x) - stale_x.detach()
            cached_outputs.append(x.cpu().detach())
            transfer_time = transfer_time + time.perf_counter()-transfer_time_start

        compute_time_start = time.perf_counter()
        x = net.linear(x)
        x = F.log_softmax(x, dim=1)
        compute_time = compute_time + time.perf_counter()-compute_time_start

        transfer_time_start = time.perf_counter()
        for ell in range(self.n_layers):
            self.hiddens[ell, sampled_nodes[ell]] = cached_outputs[ell]
        transfer_time = transfer_time + time.perf_counter()-transfer_time_start

        return x, {'compute_time': compute_time, 'transfer_time':transfer_time}

    # do not update hiddens, this is the most brilliant coding trick in this file !!!
    def forward_mini_staled(self, staled_net, x, adjs, sampled_nodes):
        transfer_time = 0
        compute_time = 0

        for ell in range(self.n_layers):
            compute_time_start = time.perf_counter()
            x = staled_net.gcs[ell](x, adjs[ell])
            x = staled_net.dropout(staled_net.relu(x))
            compute_time = compute_time + time.perf_counter()-compute_time_start

            transfer_time_start = time.perf_counter()
            x = x - x.detach() + self.hiddens[ell, sampled_nodes[ell]].to(x)
            transfer_time = transfer_time + time.perf_counter()-transfer_time_start

        compute_time_start = time.perf_counter()
        x = staled_net.linear(x)
        x = F.log_softmax(x, dim=1)
        compute_time = compute_time + time.perf_counter()-compute_time_start
        return x, {'compute_time': compute_time, 'transfer_time':transfer_time}

    def forward_full(self, net, x, adjs, sampled_nodes):
        transfer_time = 0
        compute_time = 0

        for ell in range(self.n_layers):
            compute_time_start = time.perf_counter()
            x = net.gcs[ell](x, adjs[ell])
            x = net.relu(x)
            x = net.dropout(x)
            compute_time = compute_time + time.perf_counter()-compute_time_start

            transfer_time_start = time.perf_counter()
            self.hiddens[ell, sampled_nodes[ell]] = x.cpu().detach()
            transfer_time = transfer_time + time.perf_counter()-transfer_time_start
        
        compute_time_start = time.perf_counter()
        x = net.linear(x)
        x = F.log_softmax(x, dim=1)
        compute_time = compute_time + time.perf_counter()-compute_time_start
        return x, {'compute_time': compute_time, 'transfer_time':transfer_time}


"""
SGCN++ (Doubly variance reduction)
"""

def sgcn_doubly(net, optimizer, feat_data, labels,
                train_nodes, valid_nodes, 
                adjs_full, input_nodes_full, output_nodes_full, sampled_nodes_full,
                train_data, inner_loop_num, forward_wrapper, device, dist_bound=2e-3, calculate_grad_vars=False):
    """
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    net.train()
    running_loss = []
    iter_num = 0
    grad_variance = []
    
    transfer_time = 0
    compute_time = 0  

    # record previous net full gradient
    pre_net_full = copy.deepcopy(net)
    # record previous net mini batch gradient
    pre_net_mini_v1 = copy.deepcopy(net)
    pre_net_mini_v2 = copy.deepcopy(net)

    # Run over the train_loader  
    
    run_snapshot=True
    while iter_num < inner_loop_num:
        for adjs, input_nodes, output_nodes, sampled_nodes in train_data:
            # run snapthot
            if run_snapshot:
                optimizer.zero_grad()

                outputs, time_counter = forward_wrapper.forward_full(net, feat_data[input_nodes_full], adjs_full, sampled_nodes_full)
                transfer_time += time_counter['transfer_time']
                compute_time += time_counter['compute_time']

                compute_time_start = time.perf_counter()
                current_loss = F.nll_loss(outputs, labels[output_nodes_full])
                current_loss.backward()
                compute_time = compute_time + time.perf_counter() - compute_time_start
                
                initial_hiddens = copy.deepcopy(forward_wrapper.hiddens)
                optimizer.step()
                running_loss += [current_loss.cpu().detach()]
                run_snapshot = False
            
            # record previous net full gradient
            for p_net, p_full in zip(net.parameters(), pre_net_full.parameters()):
                p_full.grad = copy.deepcopy(p_net.grad)

            # run normal 
            transfer_time_start = time.perf_counter()
            adjs = package_mxl(adjs, device)
            transfer_time = transfer_time + time.perf_counter() - transfer_time_start
            # compute previous stochastic gradient
            pre_net_mini_v1.zero_grad()
            outputs, time_counter = forward_wrapper.forward_mini_staled(
                pre_net_mini_v1, feat_data[input_nodes], adjs, sampled_nodes)
            transfer_time += time_counter['transfer_time']
            compute_time += time_counter['compute_time']

            compute_time_start = time.perf_counter()
            staled_loss = F.nll_loss(outputs, labels[output_nodes])
            staled_loss.backward()
            compute_time = compute_time + time.perf_counter() - compute_time_start

            # compute current stochastic gradient
            pre_net_mini_v2.zero_grad()
            optimizer.zero_grad()

            outputs, time_counter = forward_wrapper.forward_mini(
                net, pre_net_mini_v2, feat_data[input_nodes], adjs, sampled_nodes)
            transfer_time += time_counter['transfer_time']
            compute_time += time_counter['compute_time']

            # make sure the aggregated hiddens not too far
            current_hiddens = copy.deepcopy(forward_wrapper.hiddens)
            dist = (current_hiddens-initial_hiddens).abs().mean()
            if dist > dist_bound:
                print("RETO >>>")
                run_snapshot = True
                continue

            compute_time_start = time.perf_counter()
            current_loss = F.nll_loss(outputs, labels[output_nodes])
            current_loss.backward()
            compute_time = compute_time + time.perf_counter() - compute_time_start

            # take SCSG gradient step
            for p_net, p_mini, p_full in zip(net.parameters(), pre_net_mini_v1.parameters(), pre_net_full.parameters()):
                p_net.grad.data = p_net.grad.data + p_full.grad.data - p_mini.grad.data

            # only for experiment purpose to demonstrate ...
            if calculate_grad_vars:
                grad_variance.append(calculate_grad_variance(
                    net, feat_data, labels, train_nodes, adjs_full))

            # record previous net mini batch gradient
            for p_mini_v1, p_mini_v2, p_net in zip(pre_net_mini_v1.parameters(), pre_net_mini_v2.parameters(), net.parameters()):
                p_mini_v1.data = copy.deepcopy(p_net.data)
                p_mini_v2.data = copy.deepcopy(p_net.data)

            optimizer.step()

            # print statistics
            running_loss += [current_loss.cpu().detach()]
            iter_num += 1.0

    # calculate training loss
    train_loss = np.mean(running_loss)

    return train_loss, running_loss, grad_variance, {'compute_time': compute_time, 'transfer_time':transfer_time}


def sgcn_zeroth(net, optimizer, feat_data, labels,
                train_nodes, valid_nodes, 
                adjs_full, input_nodes_full, output_nodes_full, sampled_nodes_full,
                train_data, inner_loop_num, forward_wrapper, device, dist_bound=2e-3, calculate_grad_vars=False):
    """
    Function to updated weights with a Multi-Level SPIDER backpropagation
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    net.train()
    running_loss = []
    iter_num = 0
    grad_variance = []

    transfer_time = 0
    compute_time = 0  
    
    # Run over the train_loader
    run_snapshot = True
    while iter_num < inner_loop_num:
        for adjs, input_nodes, output_nodes, sampled_nodes in train_data:
            # run snapshot
            if run_snapshot:
                pre_net_mini = copy.deepcopy(net)
                optimizer.zero_grad()
                outputs, time_counter = forward_wrapper.forward_full(
                    net, feat_data[input_nodes_full], adjs_full, sampled_nodes_full)
                transfer_time += time_counter['transfer_time']
                compute_time += time_counter['compute_time']

                compute_time_start = time.perf_counter()
                current_loss = F.nll_loss(outputs, labels[output_nodes_full])
                current_loss.backward()
                compute_time = compute_time + time.perf_counter() - compute_time_start

                initial_hiddens = copy.deepcopy(forward_wrapper.hiddens)
                optimizer.step()
                running_loss += [current_loss.cpu().detach()]
                run_snapshot = False
            
            # run 
            transfer_time_start = time.perf_counter()
            adjs = package_mxl(adjs, device)
            transfer_time = transfer_time + time.perf_counter() - transfer_time_start
            # compute previous stochastic gradient and compute current stochastic gradient
            optimizer.zero_grad()

            outputs, time_counter = forward_wrapper.forward_mini(
                net, pre_net_mini, feat_data[input_nodes], adjs, sampled_nodes)
            transfer_time += time_counter['transfer_time']
            compute_time += time_counter['compute_time']

            # make sure the aggregated hiddens not too far
            current_hiddens = copy.deepcopy(forward_wrapper.hiddens)
            dist = (current_hiddens-initial_hiddens).abs().mean()
            if dist > dist_bound:
                run_snapshot = True
                print("RETO >>>")
                continue

            compute_time_start = time.perf_counter()
            current_loss = F.nll_loss(outputs, labels[output_nodes])
            current_loss.backward()
            compute_time = compute_time + time.perf_counter() - compute_time_start

            # record previous net mini batch gradient
            for p_mini, p_net in zip(pre_net_mini.parameters(), net.parameters()):
                p_mini.data = copy.deepcopy(p_net.data)

            # only for experiment purpose to demonstrate ...
            if calculate_grad_vars:
                grad_variance.append(calculate_grad_variance(
                    net, feat_data, labels, train_nodes, adjs_full))
            optimizer.step()

            # print statistics
            running_loss += [current_loss.cpu().detach()]
            iter_num += 1.0

    # calculate training loss
    train_loss = np.mean(running_loss)

    return train_loss, running_loss, grad_variance, {'compute_time': compute_time, 'transfer_time':transfer_time}

"""
VRGCN wrapper
"""

class VRGCNWrapper(nn.Module):
    def __init__(self, n_nodes, n_hid, n_layers, n_classes):
        super(VRGCNWrapper, self).__init__()
        self.n_layers = n_layers
        self.hiddens = torch.zeros(n_layers, n_nodes, n_hid)

    def forward_full(self, net, x, adjs, sampled_nodes):
        transfer_time = 0
        compute_time = 0

        for ell in range(len(net.gcs)):
            compute_time_start = time.perf_counter()
            x_ = net.gcs[ell](x, adjs[ell])
            x = net.relu(x_)
            x = net.dropout(x)
            compute_time = compute_time + time.perf_counter()-compute_time_start

            transfer_time_start = time.perf_counter()
            self.hiddens[ell,sampled_nodes[ell]] = x_.cpu().detach()
            transfer_time = transfer_time + time.perf_counter()-transfer_time_start
            
        compute_time_start = time.perf_counter()
        x = net.linear(x)
        x = F.log_softmax(x, dim=1)
        compute_time = compute_time + time.perf_counter()-compute_time_start
        return x, {'compute_time': compute_time, 'transfer_time':transfer_time}

    def forward_mini(self, net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes):
        transfer_time = 0
        compute_time = 0

        cached_outputs = []
        for ell in range(len(net.gcs)):
            transfer_time_start = time.perf_counter()
            x_bar = x if ell == 0 else net.dropout(net.relu(self.hiddens[ell-1,sampled_nodes[ell-1]].to(x)))
            x_bar_exact = x_exact[input_exact_nodes[ell]] if ell == 0 else net.dropout(net.relu(self.hiddens[ell-1,input_exact_nodes[ell]].to(x)))
            transfer_time = transfer_time + time.perf_counter()-transfer_time_start
            
            compute_time_start = time.perf_counter()
            x_ = net.gcs[ell](x, adjs[ell]) - net.gcs[ell](x_bar, adjs[ell]) + net.gcs[ell](x_bar_exact, adjs_exact[ell])
            x = net.relu(x_)
            x = net.dropout(x)
            compute_time = compute_time + time.perf_counter()-compute_time_start

            transfer_time_start = time.perf_counter()
            cached_outputs += [x_.detach().cpu()]
            transfer_time = transfer_time + time.perf_counter()-transfer_time_start

        compute_time_start = time.perf_counter()
        x = net.linear(x)
        x = F.log_softmax(x, dim=1)
        compute_time = compute_time + time.perf_counter()-compute_time_start
    
        transfer_time_start = time.perf_counter()
        for ell in range(len(net.gcs)):
            self.hiddens[ell, sampled_nodes[ell]] = cached_outputs[ell]
        transfer_time = transfer_time + time.perf_counter()-transfer_time_start
        return x, {'compute_time': compute_time, 'transfer_time':transfer_time}
    
    # do not update hiddens, this is the most brilliant coding trick in this file !!!
    def forward_mini_staled(self, net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes):
        transfer_time = 0
        compute_time = 0

        cached_outputs = []
        for ell in range(len(net.gcs)):
            transfer_time_start = time.perf_counter()
            x_bar = x if ell == 0 else net.dropout(net.relu(self.hiddens[ell-1,sampled_nodes[ell-1]].to(x)))
            x_bar_exact = x_exact[input_exact_nodes[ell]] if ell == 0 else net.dropout(net.relu(self.hiddens[ell-1,input_exact_nodes[ell]].to(x)))
            transfer_time = transfer_time + time.perf_counter()-transfer_time_start

            compute_time_start = time.perf_counter()
            x_ = net.gcs[ell](x, adjs[ell]) - net.gcs[ell](x_bar, adjs[ell]) + net.gcs[ell](x_bar_exact, adjs_exact[ell])
            x = net.relu(x_)
            x = net.dropout(x)
            compute_time = compute_time + time.perf_counter()-compute_time_start

            transfer_time_start = time.perf_counter()
            cached_outputs += [x_.detach().cpu()]
            transfer_time = transfer_time + time.perf_counter()-transfer_time_start

        compute_time_start = time.perf_counter()
        x = net.linear(x)
        x = F.log_softmax(x, dim=1)
        compute_time = compute_time + time.perf_counter()-compute_time_start
        return x, {'compute_time': compute_time, 'transfer_time':transfer_time}
        
    def partial_grad(self, net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes, targets, weight=None):
        transfer_time = 0
        compute_time = 0

        compute_time_start = time.perf_counter()
        outputs, time_counter = self.forward_mini(net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes)
        compute_time = compute_time + time_counter['compute_time']
        transfer_time = transfer_time + time_counter['transfer_time']
        
        loss = F.nll_loss(outputs, targets)
        loss.backward()
        compute_time = compute_time + time.perf_counter()-compute_time_start
        return loss.detach(), {'compute_time': compute_time, 'transfer_time':transfer_time}
    
    def partial_grad_staled(self, net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes, targets, weight=None):
        transfer_time = 0
        compute_time = 0

        compute_time_start = time.perf_counter()
        outputs, time_counter = self.forward_mini_staled(net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes)
        compute_time = compute_time + time_counter['compute_time']
        transfer_time = transfer_time + time_counter['transfer_time']

        loss = F.nll_loss(outputs, targets)
        loss.backward()
        compute_time = compute_time + time.perf_counter()-compute_time_start
        return loss.detach(), {'compute_time': compute_time, 'transfer_time':transfer_time}
    
def VRGCN_step(net, optimizer, feat_data, labels,
             train_nodes, valid_nodes,
             adjs_full, train_data, inner_loop_num, wrapper, device, calculate_grad_vars=False):
    """
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    transfer_time = 0
    compute_time = 0  

    net.train()

    running_loss = []
    iter_num = 0.0

    grad_variance = []
    # Run over the train_loader
    while iter_num < inner_loop_num:
        for adjs, adjs_exact, input_nodes, output_nodes, sampled_nodes, input_exact_nodes in train_data:
            transfer_time_start = time.perf_counter()
            adjs = package_mxl(adjs, device)
            adjs_exact = package_mxl(adjs_exact, device)
            transfer_time = transfer_time + time.perf_counter() - transfer_time_start
            # compute current stochastic gradient
            optimizer.zero_grad()

            current_loss, time_counter = wrapper.partial_grad(net, 
                feat_data[input_nodes], adjs, sampled_nodes, feat_data, adjs_exact, input_exact_nodes, labels[output_nodes])
            transfer_time += time_counter['transfer_time']
            compute_time += time_counter['compute_time']

            # only for experiment purpose to demonstrate ...
            if calculate_grad_vars:
                grad_variance.append(calculate_grad_variance(
                    net, feat_data, labels, train_nodes, adjs_full))

            optimizer.step()

            # print statistics
            running_loss += [current_loss.cpu().detach()]
            iter_num += 1.0

    # calculate training loss
    train_loss = np.mean(running_loss)

    return train_loss, running_loss, grad_variance, {'compute_time': compute_time, 'transfer_time':transfer_time}
    
def VRGCN_doubly(net, optimizer, feat_data, labels,
                 train_nodes, valid_nodes,
                 adjs_full, train_data, inner_loop_num, wrapper, device, calculate_grad_vars=False):
    """
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    transfer_time = 0
    compute_time = 0  

    net.train()
    # record previous net full gradient
    pre_net_full = copy.deepcopy(net)
    # record previous net mini batch gradient
    pre_net_mini = copy.deepcopy(net)
    
    # Compute full grad
    optimizer.zero_grad()
    current_loss, _ = net.calculate_loss_grad(
        feat_data, adjs_full, labels, train_nodes)
    optimizer.step()
    
    running_loss = []
    iter_num = 0.0

    grad_variance = []
    # Run over the train_loader
    while iter_num < inner_loop_num:
        for adjs, adjs_exact, input_nodes, output_nodes, sampled_nodes, input_exact_nodes in train_data:
            transfer_time_start = time.perf_counter()
            adjs = package_mxl(adjs, device)
            adjs_exact = package_mxl(adjs_exact, device)
            transfer_time = transfer_time + time.perf_counter() - transfer_time_start
            # record previous net full gradient
            for p_net, p_full in zip(net.parameters(), pre_net_full.parameters()):
                p_full.grad = copy.deepcopy(p_net.grad)
                
            # compute previous stochastic gradient
            pre_net_mini.zero_grad()
            # take backward
            _, time_counter = wrapper.partial_grad_staled(pre_net_mini, 
                feat_data[input_nodes], adjs, sampled_nodes, feat_data, adjs_exact, input_exact_nodes, labels[output_nodes])
            transfer_time += time_counter['transfer_time']
            compute_time += time_counter['compute_time']

            # compute current stochastic gradient
            optimizer.zero_grad()

            current_loss, time_counter = wrapper.partial_grad(net, 
                feat_data[input_nodes], adjs, sampled_nodes, feat_data, adjs_exact, input_exact_nodes, labels[output_nodes])
            transfer_time += time_counter['transfer_time']
            compute_time += time_counter['compute_time']

            # take SCSG gradient step
            for p_net, p_mini, p_full in zip(net.parameters(), pre_net_mini.parameters(), pre_net_full.parameters()):
                p_net.grad.data = p_net.grad.data + p_full.grad.data - p_mini.grad.data
                
            # only for experiment purpose to demonstrate ...
            if calculate_grad_vars:
                grad_variance.append(calculate_grad_variance(
                    net, feat_data, labels, train_nodes, adjs_full))

            # record previous net mini batch gradient
            for p_mini, p_net in zip(pre_net_mini.parameters(), net.parameters()):
                p_mini.data = copy.deepcopy(p_net.data)
                
            optimizer.step()

            # print statistics
            running_loss += [current_loss.cpu().detach()]
            iter_num += 1.0

    # calculate training loss
    train_loss = np.mean(running_loss)

    return train_loss, running_loss, grad_variance, {'compute_time': compute_time, 'transfer_time':transfer_time}