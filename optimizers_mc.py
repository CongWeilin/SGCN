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
SPIDER wrapper
"""


def spider_step(net, optimizer, feat_data, labels,
                train_nodes, valid_nodes,
                adjs_full, train_data, inner_loop_num, device, calculate_grad_vars=False):
    """
    Function to updated weights with a SPIDER backpropagation
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
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

    running_loss = [current_loss.cpu().detach()]
    iter_num = 0

    grad_variance = []
    # Run over the train_loader
    while iter_num < inner_loop_num:
        for adjs, input_nodes, output_nodes, sampled_nodes in train_data:
            adjs = package_mxl(adjs, device)

            # record previous net full gradient
            for p_net, p_full in zip(net.parameters(), pre_net_full.parameters()):
                p_full.grad = copy.deepcopy(p_net.grad)

            # compute previous stochastic gradient
            pre_net_mini.zero_grad()
            # take backward
            pre_net_mini.partial_grad(
                feat_data[input_nodes], adjs, labels[output_nodes])

            # compute current stochastic gradient
            optimizer.zero_grad()
            current_loss = net.partial_grad(
                feat_data[input_nodes], adjs, labels[output_nodes])

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

    return train_loss, running_loss, grad_variance


"""
SGD wrapper
"""


def sgd_step(net, optimizer, feat_data, labels,
             train_nodes, valid_nodes,
             adjs_full, train_data, inner_loop_num, device, calculate_grad_vars=False):
    """
    Function to updated weights with a SGD backpropagation
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    net.train()

    running_loss = []
    iter_num = 0.0

    grad_variance = []
    # Run over the train_loader
    while iter_num < inner_loop_num:
        for adjs, input_nodes, output_nodes, sampled_nodes in train_data:
            adjs = package_mxl(adjs, device)

            # compute current stochastic gradient
            optimizer.zero_grad()
            current_loss = net.partial_grad(
                feat_data[input_nodes], adjs, labels[output_nodes])

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

    return train_loss, running_loss, grad_variance


"""
Full-batch
"""


def full_step(net, optimizer, feat_data, labels,
              train_nodes, valid_nodes,
              adjs_full, train_data, inner_loop_num, device, calculate_grad_vars=False):
    """
    Function to updated weights with a SGD backpropagation
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    net.train()

    running_loss = []
    iter_num = 0.0

    grad_variance = []
    # Run over the train_loader
    while iter_num < inner_loop_num:

            # compute current stochastic gradient
        optimizer.zero_grad()
        current_loss, _ = net.calculate_loss_grad(
            feat_data, adjs_full, labels, train_nodes)

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

    return train_loss, running_loss, grad_variance


"""
Used for Multi-level Spider
"""


class ForwardWrapper(nn.Module):
    def __init__(self, n_nodes, n_hid, n_layers, n_classes):
        super(ForwardWrapper, self).__init__()
        self.n_layers = n_layers
        self.hiddens = torch.zeros(n_layers, n_nodes, n_hid)

    def forward_mini(self, net, staled_net, x, adjs, sampled_nodes):
        cached_outputs = []
        for ell in range(self.n_layers):
            stale_x = x if ell == 0 else self.hiddens[ell -
                                                      1, sampled_nodes[ell-1]].to(x)
            stale_x = staled_net.gcs[ell](stale_x, adjs[ell])
            stale_x = staled_net.dropout(staled_net.relu(stale_x))
            x = net.gcs[ell](x, adjs[ell])
            x = net.dropout(net.relu(x))
            x = x + self.hiddens[ell, sampled_nodes[ell]
                                 ].to(x) - stale_x.detach()
            cached_outputs.append(x.cpu().detach())
        x = net.linear(x)
        x = torch.sigmoid(x)
        for ell in range(self.n_layers):
            self.hiddens[ell, sampled_nodes[ell]] = cached_outputs[ell]
        return x

    # do not update hiddens, this is the most brilliant coding trick in this file !!!
    def forward_mini_staled(self, staled_net, x, adjs, sampled_nodes):
        for ell in range(self.n_layers):
            x = staled_net.gcs[ell](x, adjs[ell])
            x = staled_net.dropout(staled_net.relu(x))
            x = x - x.detach() + self.hiddens[ell, sampled_nodes[ell]].to(x)
        x = staled_net.linear(x)
        x = torch.sigmoid(x)
        return x

    def forward_full(self, net, x, adjs, sampled_nodes):
        for ell in range(self.n_layers):
            x = net.gcs[ell](x, adjs[ell])
            x = net.relu(x)
            x = net.dropout(x)
            self.hiddens[ell, sampled_nodes[ell]] = x.cpu().detach()
        x = net.linear(x)
        x = torch.sigmoid(x)
        return x


"""
Multi-level Variance Reduction
"""


def multi_level_step_v2(net, optimizer, feat_data, labels,
                               train_nodes, valid_nodes, adjs_full, sampled_nodes_full,
                               train_data, inner_loop_num, forward_wrapper, device, dist_bound=1e-3, calculate_grad_vars=False):
    """
    Function to updated weights with a Multi-Level SPIDER backpropagation
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    net.train()
    running_loss = []
    iter_num = 0
    grad_variance = []

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
#                 print('run snapshot')
                optimizer.zero_grad()
                outputs = forward_wrapper.forward_full(
                    net, feat_data, adjs_full, sampled_nodes_full)
                current_loss = F.binary_cross_entropy(outputs[train_nodes], labels[train_nodes])
                current_loss.backward()
                
                initial_hiddens = copy.deepcopy(forward_wrapper.hiddens)
                optimizer.step()
                running_loss += [current_loss.cpu().detach()]
                run_snapshot = False
                
            # run
            adjs = package_mxl(adjs, device)

            # record previous net full gradient
            for p_net, p_full in zip(net.parameters(), pre_net_full.parameters()):
                p_full.grad = copy.deepcopy(p_net.grad)

            # compute previous stochastic gradient
            pre_net_mini_v1.zero_grad()
            outputs = forward_wrapper.forward_mini_staled(
                pre_net_mini_v1, feat_data[input_nodes], adjs, sampled_nodes)
            staled_loss = F.binary_cross_entropy(outputs, labels[output_nodes])
            staled_loss.backward()

            # compute current stochastic gradient
            pre_net_mini_v2.zero_grad()
            optimizer.zero_grad()

            outputs = forward_wrapper.forward_mini(
                net, pre_net_mini_v2, feat_data[input_nodes], adjs, sampled_nodes)

            # make sure the aggregated hiddens not too far
            current_hiddens = copy.deepcopy(forward_wrapper.hiddens)
            dist = (current_hiddens-initial_hiddens).abs().mean()
            if dist > dist_bound:
                print("RETO >>>")
                run_snapshot = True
                continue
                
            current_loss = F.binary_cross_entropy(
                outputs, labels[output_nodes])
            current_loss.backward()

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

    return train_loss, running_loss, grad_variance


def multi_level_step_v1(net, optimizer, feat_data, labels,
                               train_nodes, valid_nodes, adjs_full, sampled_nodes_full,
                               train_data, inner_loop_num, forward_wrapper, device, dist_bound=1e-3, calculate_grad_vars=False):
    """
    Function to updated weights with a Multi-Level SPIDER backpropagation
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    net.train()
    running_loss = []
    iter_num = 0
    grad_variance = []

    # Run over the train_loader
    run_snapshot = True
    while iter_num < inner_loop_num:
        for adjs, input_nodes, output_nodes, sampled_nodes in train_data:
            # run snapshot
            if run_snapshot:
                pre_net_mini = copy.deepcopy(net)
                optimizer.zero_grad()
                outputs = forward_wrapper.forward_full(
                    net, feat_data, adjs_full, sampled_nodes_full)
                current_loss = F.binary_cross_entropy(outputs[train_nodes], labels[train_nodes])
                current_loss.backward()
                initial_hiddens = copy.deepcopy(forward_wrapper.hiddens)
                optimizer.step()
                running_loss += [current_loss.cpu().detach()]
                run_snapshot = False
                
            # run
            adjs = package_mxl(adjs, device)
            # compute previous stochastic gradient and compute current stochastic gradient
            optimizer.zero_grad()

            outputs = forward_wrapper.forward_mini(
                net, pre_net_mini, feat_data[input_nodes], adjs, sampled_nodes)

            # make sure the aggregated hiddens not too far
            current_hiddens = copy.deepcopy(forward_wrapper.hiddens)
            dist = (current_hiddens-initial_hiddens).abs().mean()
            if dist > dist_bound:
                run_snapshot = True
                print("RETO >>>")
                continue

            current_loss = F.binary_cross_entropy(
                outputs, labels[output_nodes])
            current_loss.backward()

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

    return train_loss, running_loss, grad_variance


"""
Used for Multi-level Momentum
"""


class ForwardWrapperMomentum(nn.Module):
    def __init__(self, n_nodes, n_hid, n_layers, n_classes):
        super(ForwardWrapperMomentum, self).__init__()
        self.n_layers = n_layers
        self.n_times = torch.zeros(n_layers, n_nodes, 1)
        self.hiddens = torch.zeros(n_layers, n_nodes, n_hid)

    def calculate_gamma(self, ell, sampled_nodes, c=0.5):
        num_steps = self.n_times[ell, sampled_nodes[ell]]
        mask = num_steps < 2
        num_steps[mask] = 2  # to get rid of the warning 1/0
        gamma = c * num_steps / (1.0 + c * num_steps)
        gamma *= 1.0 - torch.sqrt((1.0 - c) /
                                  (num_steps * (num_steps + 1))) / c
        gamma[mask] = 0
        return gamma

    def forward_mini(self, net, x, adjs, sampled_nodes):
        cached_outputs = []
        for ell in range(self.n_layers):
            x = net.gcs[ell](x, adjs[ell])
            x = net.dropout(net.relu(x))
            gamma = self.calculate_gamma(ell, sampled_nodes)
            gamma = gamma.to(x)
            x = gamma*x + (1-gamma)*self.hiddens[ell, sampled_nodes[ell]].to(x)
            cached_outputs.append(x.cpu().detach())
        x = net.linear(x)
        x = torch.sigmoid(x)
        for ell in range(self.n_layers):
            self.hiddens[ell, sampled_nodes[ell]] = cached_outputs[ell]
            self.n_times[ell, sampled_nodes[ell]] += 1
        return x


def multi_level_momentum_step(net, optimizer, feat_data, labels,
                              train_nodes, valid_nodes, adjs_full, sampled_nodes_full,
                              train_data, inner_loop_num, forward_wrapper, device, calculate_grad_vars=False):
    """
    Function to updated weights with a Multi-Level SPIDER backpropagation
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    net.train()

    running_loss = []
    iter_num = 0

    grad_variance = []
    # Run over the train_loader
    while iter_num < inner_loop_num:
        for adjs, input_nodes, output_nodes, sampled_nodes in train_data:
            adjs = package_mxl(adjs, device)
            # compute previous stochastic gradient and compute current stochastic gradient
            optimizer.zero_grad()

            outputs = forward_wrapper.forward_mini(
                net, feat_data[input_nodes], adjs, sampled_nodes)

            current_loss = F.binary_cross_entropy(
                outputs, labels[output_nodes])
            current_loss.backward()

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

    return train_loss, running_loss, grad_variance
