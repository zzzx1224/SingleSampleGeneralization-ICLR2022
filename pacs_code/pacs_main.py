from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pdb
import os, shutil
import argparse
import time
from tensorboardX import SummaryWriter
from aug import *
import pdb
from pacs_rtdataset import *
from pacs_dataset import *
import pacs_model
import sys

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='learning rate')
parser.add_argument('--sparse', default=0, type=float, help='L1 panelty')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log1', help='Log dir [default: log]')
parser.add_argument('--dataset', default='PACS', help='datasets')
parser.add_argument('--batch_size', type=int, default=128, help='Batch Size during training [default: 32]')
parser.add_argument('--bases', type=int, default=7, help='Batch Size during training [default: 32]')
parser.add_argument('--shuffle', type=int, default=0, help='Batch Size during training [default: 32]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--sharing', default='layer', help='Log dir [default: log]')
parser.add_argument('--net', default='res18', help='res18 or res50')
parser.add_argument('--l2', action='store_true')
parser.add_argument('--base', action='store_true')
parser.add_argument('--autodecay', action='store_true')
parser.add_argument('--share_bases', action='store_true')
parser.add_argument('--hychy', type=int, default=0, help='hyrarchi')
parser.add_argument('--sub', default=1.0, type=float, help='subset of tinyimagenet')
parser.add_argument('--test_domain', default='sketch', help='GPU to use [default: GPU 0]')
parser.add_argument('--train_domain', default='', help='GPU to use [default: GPU 0]')
parser.add_argument('--ite_train', default=True, type=bool, help='learning rate')
parser.add_argument('--max_ite', default=10000, type=int, help='max_ite')
parser.add_argument('--test_ite', default=50, type=int, help='learning rate')
parser.add_argument('--bias', default=1, type=int, help='whether sample')
parser.add_argument('--test_batch', default=100, type=int, help='learning rate')
parser.add_argument('--data_aug', default=1, type=int, help='whether sample')
parser.add_argument('--difflr', default=1, type=int, help='whether sample')
parser.add_argument('--mc_times', default=10, type=int, help='number of Monte Carlo samples')
parser.add_argument('--mbeta', default=1e-3, type=float, help='beta for mid y')
parser.add_argument('--abeta', default=1e-5, type=float, help='beta for adaptive kl')
parser.add_argument('--alpha', default=0.5, type=float, help='beta for adaptive kl')
parser.add_argument('--norm', default='bn', help='bn or in')
parser.add_argument('--domain_l', default=4, type=int, help='1 or 3 or 4')
parser.add_argument('--test_sample', default=False, type=bool, help='sampling in test time')
parser.add_argument('--dinit', default='rt', help='random r or feature f or center feature c')
parser.add_argument('--reslr', default=0.5, type=float, help='backbone learning rate')
parser.add_argument('--pbeta', default=1, type=float, help='backbone learning rate')
parser.add_argument('--agg_model', default='concat', help='concat or bayes or rank1')
parser.add_argument('--agg_method', default='mean', help='ensemble or mean or ronly')
parser.add_argument('--dom_sta', default='mean', help='both or mean')
parser.add_argument('--ptest', default=0, type=int, help='use prior in test')
parser.add_argument('--sharem', default=0, type=int, help='share model or not')
parser.add_argument('--ctx_num', default=10, type=int, help='learning rate')
parser.add_argument('--hierar', default=2, type=int, help='hierarchical model')
parser.add_argument('--adp_num', default='1', help='learning rate')
parser.add_argument('--mul_tra_sam', default=0, type=int, help='hierarchical model')
parser.add_argument('--dsgrad', default=1, type=int, help='whether sample')

args = parser.parse_args()
BATCH_SIZE = args.batch_size
OPTIMIZER = args.optimizer
gpu_index = args.gpu
backbone = args.net
dsgrad = bool(args.dsgrad)
max_ite = args.max_ite
test_ite = args.test_ite
test_batch = args.test_batch
iteration_training = args.ite_train
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
test_domain = args.test_domain
train_domain = args.train_domain
num_adp = args.adp_num
mid_beta = args.mbeta
ctx_num = args.ctx_num
a_beta = args.abeta
multi_train_sam = bool(args.mul_tra_sam)
p_beta = args.pbeta
ifsample = args.test_sample
Dinit = args.dinit
dom_sta = args.dom_sta
difflr = args.difflr
res_lr = args.reslr
hierar = args.hierar
agg_model = args.agg_model
agg_method = args.agg_method
prior_test = args.ptest
prior_test = bool(prior_test)
with_bias = args.bias
with_bias = bool(with_bias)
difflr = bool(difflr)
sharemodel = bool(args.sharem)
alpha = args.alpha

norm_method = args.norm
zdlayers = args.domain_l

mc_times = args.mc_times
# ifcommon = args.ifcommon
# ifadapt = args.ifadapt

data_aug = args.data_aug
data_aug = bool(data_aug)

LOG_DIR = os.path.join('logs', args.log_dir)
args.log_dir = LOG_DIR

name_file = sys.argv[0]
if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
os.mkdir(LOG_DIR)
os.mkdir(LOG_DIR + '/train_img')
os.mkdir(LOG_DIR + '/test_img')
os.mkdir(LOG_DIR + '/files')
os.system('cp %s %s' % (name_file, LOG_DIR))
os.system('cp %s %s' % ('*.py', os.path.join(LOG_DIR, 'files')))
os.system('cp -r %s %s' % ('models', os.path.join(LOG_DIR, 'files')))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
print(args)
LOG_FOUT.write(str(args)+'\n')


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-2)
            if m.bias is not None:
                init.constant(m.bias, 0)


def log_string(out_str, print_out=True):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    if print_out:
        print(out_str)

st = ' '
log_string(st.join(sys.argv))

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_valid_acc = 0 # best validation accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


writer = SummaryWriter(log_dir=args.log_dir)

# Data
print('==> Preparing data..')

bird = False

decay_inter = [250, 450]

if args.dataset == 'PACS':
    NUM_CLASS = 7
    num_domain = 4
    batchs_per_epoch = 0
    # ctx_test = 2 * ctx_num
    ctx_test = ctx_num
    domains = ['art_painting', 'photo', 'cartoon', 'sketch']
    assert test_domain in domains
    domains.remove(test_domain)
    if train_domain:
        domains = train_domain.split(',')
    log_string('data augmentation is ' + str(data_aug))
    if data_aug:
        # log_string()
        transform_train = transforms.Compose([
            # transforms.RandomCrop(64, padding=4),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(0.75, 1.33), interpolation=2),
            transforms.RandomHorizontalFlip(),
            ImageJitter(jitter_param),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    log_string('train_domain: ' + str(domains))
    log_string('test: ' + str(test_domain))
    
    all_dataset = PACS(test_domain)
    rt_context = rtPACS(test_domain, ctx_num)
else:
    raise NotImplementedError

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

args.num_classes = NUM_CLASS
args.num_domains = num_domain
args.bird = bird

# Model
print('==> Building model..')

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


net = pacs_model.net0(args.num_classes, len(domains), backbone, Dinit, agg_model, agg_method, dsgrad, zdlayers, dom_sta, prior_test, hierar, num_adp, multi_train_sam, sharemodel)

net.apply(inplace_relu)

if Dinit=='c':
    contexts = []
    for training_domain in domains:
        cf = np.load('centf_'+training_domain+'.npy')
        # pdb.set_trace()
        cf = torch.Tensor(cf).cuda()
        contexts.append(cf)
    net.context_init(contexts)

# pdb.set_trace()

def convert_bn_2_in(model):
    for n, v in model.named_children():
        if isinstance(v, nn.BatchNorm2d):
            # pdb.set_trace()
            w = v.weight.data
            b = v.weight.data
            num_channels = w.shape[0]
            setattr(model, n, nn.InstanceNorm2d(num_channels, affine=True))
            getattr(model, n).weight.data = w
            getattr(model, n).bias.data = b

        else:
            convert_bn_2_in(v)

if args.norm=='in':
    convert_bn_2_in(net)
# pdb.set_trace()

log_string(str(net.extra_repr))

pc = get_parameter_number(net)
log_string('Total: %.4fM, Trainable: %.4fM' %(pc['Total']/float(1e6), pc['Trainable']/float(1e6)))

net = net.to(device)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)

# if isinstance(net,torch.nn.DataParallel):
#     net = net.module
net.train()
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# pdb.set_trace()

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

WEIGHT_DECAY = args.weight_decay

if OPTIMIZER == 'momentum':
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY, momentum=0.9)
elif OPTIMIZER == 'nesterov':
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY, momentum=0.9, nesterov=True)
elif OPTIMIZER=='adam' and difflr and agg_model=='concat' and hierar==0:
    optimizer = torch.optim.Adam([{'params': net.layer0.parameters(), 'lr':args.lr * res_lr},   # different lr
                                  {'params': net.layer1.parameters(), 'lr':args.lr * res_lr},
                                  {'params': net.layer2.parameters(), 'lr':args.lr * res_lr},
                                  {'params': net.layer3.parameters(), 'lr':args.lr * res_lr},
                                  {'params': net.layer4.parameters(), 'lr':args.lr * res_lr},
                                  # {'params': net.com_classifier.parameters()},
                                  {'params': net.domain_layer.parameters()},
                                  {'params': net.agg_layer.parameters()},
                                  {'params': net.shared_net.parameters()}], 
                                  lr=args.lr, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER=='adam' and difflr and agg_model=='concat' and hierar==2:
    optimizer = torch.optim.Adam([{'params': net.layer0.parameters(), 'lr':args.lr * res_lr},   # different lr
                                  {'params': net.layer1.parameters(), 'lr':args.lr * res_lr},
                                  {'params': net.layer2.parameters(), 'lr':args.lr * res_lr},
                                  {'params': net.layer3.parameters(), 'lr':args.lr * res_lr},
                                  {'params': net.layer4.parameters(), 'lr':args.lr * res_lr},
                                  # {'params': net.com_classifier.parameters()},
                                  {'params': net.domain_layer.parameters()},
                                  {'params': net.agg_layer.parameters()},
                                  {'params': net.shared_net.parameters()},
                                  {'params': net.shared_net1.parameters()}], 
                                  lr=args.lr, weight_decay=WEIGHT_DECAY)
    # print(optimizer)
elif OPTIMIZER=='adam' and not difflr:
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER == 'rmsp':
    optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
else:
    raise NotImplementedError

# pdb.set_trace()

bases_list = [b for a, b in net.named_parameters() if a.endswith('bases')]
other_list = [b for a, b in net.named_parameters() if 'coef' not in a]

coef_list = [b for a, b in net.named_parameters() if 'coef' in a]
print([a for a, b in net.named_parameters() if 'coef' in a])
print([b.shape for a, b in net.named_parameters() if 'coef' in a])
log_string('Totally %d coefs.' %(len(coef_list)))

# global converge_count 
converge_count = 0

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def train(epoch):
    log_string('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    closs = 0
    aloss = 0
    ploss = 0
    hkl = 0
    akl = 0
    correct = 0
    ac_correct = []
    total = 0
    ac_correct = [0, 0, 0]

    if epoch<3:
        domain_id = epoch
        loss_rate = 1e-8
    else:
        domain_id = np.random.randint(len(domains))
        loss_rate = 1
    print(domain_id)
    num_preds = 1
    t0 = time.time()
    all_dataset.reset('train', domain_id, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(all_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False, worker_init_fn=worker_init_fn)
    # print(time.time()-t0)
    rt_context.reset('train', transform=transform_train)
    context_loader = torch.utils.data.DataLoader(rt_context, batch_size=(num_domain-1)*NUM_CLASS*ctx_num, shuffle=False, num_workers=4, drop_last=False, worker_init_fn=worker_init_fn)
    # print(time.time()-t0)
    for batch_idx, (inputs, targets) in enumerate(context_loader):
        context_img, context_label = inputs.to(device), targets.to(device)
    # pdb.set_trace()
    # print(time.time()-t0)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # print(time.time()-t0)

        optimizer.zero_grad()

        ys, yps, adapt_kl, hkld = net(inputs, context_img, domain_id, ctx_num)

        targets_samples = targets.unsqueeze(1).repeat(1, 10).view(-1)
        results = []

        prior_loss = criterion(yps.view(-1, args.num_classes), targets_samples)
        adapt_loss = 0
        ys = ys.view(BATCH_SIZE, mc_times, num_preds, args.num_classes)
        # pdb.set_trace()
        for i in range(num_preds):
            results.append(ys[:,:,i].mean(1))
            adapt_loss += criterion(ys[:,:,i].view(-1, args.num_classes), targets_samples)

        mean_results = torch.cat(results, 0).view(num_preds, BATCH_SIZE, args.num_classes).mean(0)
        
        loss = adapt_loss + p_beta*prior_loss*int(agg_model!='bbase') + a_beta * adapt_kl + mid_beta * hkld
        # print(time.time()-t0)

        if Dinit=='f':
            loss = loss * loss_rate

        train_loss += loss.item()
        # closs += ifcommon * common_loss.item()
        aloss += adapt_loss.item()
        ploss += p_beta * prior_loss.item()*int(agg_model!='bbase')
        hkl += mid_beta * hkld.item()
        akl += a_beta * adapt_kl.item()

        loss.backward()
        optimizer.step()
        # print(time.time()-t0)

        if args.dinit != 'rt':
            net.context_update(epoch, targets, alpha)
        # pdb.set_trace()
        # print(time.time()-t0)

        # predicted = []
        for i in range(len(results)):
            _, predicted = results[i].max(1)
            ac_correct[i] += predicted.eq(targets).sum().item()
        _, mean_preditcted = mean_results.max(1)
        # pdb.set_trace()
        # print(time.time()-t0)
        correct += mean_preditcted.eq(targets).sum().item()
        total += targets.size(0)
        # print(time.time()-t0)


        if iteration_training and batch_idx>=batchs_per_epoch:
            break

    log_string('Loss: %.3f  | a_loss: %3f | p_loss: %3f | a_kl: %3f | mid_loss: %3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), aloss/(batch_idx+1), ploss/(batch_idx+1), akl/(batch_idx+1), hkl/(batch_idx+1), 100.*correct/total, correct, total))

    writer.add_scalar('cls_loss', train_loss/(batch_idx+1), epoch)
    writer.add_scalar('cls_acc', 100.*correct/total, epoch)
    print(time.time()-t0)

def validation(epoch):
    global best_valid_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    ac_correct = [0, 0, 0]
        
    # all_dataset.reset('val', 0, transform=transform_test)
    # valloader = torch.utils.data.DataLoader(all_dataset, batch_size=test_batch, shuffle=False, num_workers=4)
    rt_context.reset('val', transform=transform_test)
    context_loader = torch.utils.data.DataLoader(rt_context, batch_size=(num_domain-1)*NUM_CLASS*ctx_test, shuffle=False, num_workers=4, drop_last=False, worker_init_fn=worker_init_fn)
    for batch_idx, (inputs, targets) in enumerate(context_loader):
        context_img, context_label = inputs.to(device), targets.to(device)

    
    with torch.no_grad():
        for i in range(4):
            all_dataset.reset('val', i, transform=transform_test)
            valloader = torch.utils.data.DataLoader(all_dataset, batch_size=test_batch, shuffle=False, num_workers=4)

            num_preds = 1

            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                ys, _, _, _ = net(inputs, context_img, i, ctx_test, ifsample)

                ys = ys.view(ys.size()[0], -1, args.num_classes)

                y = torch.softmax(ys, -1).mean(1)
                
                cls_loss = criterion(y, targets)
                loss = cls_loss

                test_loss += loss.item()
                _, predicted = y.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        log_string('VAL Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        writer.add_scalar('val_loss', test_loss/(batch_idx+1), epoch)
        writer.add_scalar('val_acc', 100.*correct/total, epoch)
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_valid_acc:
        print('Saving..')
        log_string('The best validation Acc')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, os.path.join(LOG_DIR, 'ckpt.t7'))
        best_valid_acc = acc
        return 0
    else:
        return 1

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    ac_correct = [0, 0, 0, 0]

    num_preds = 1

    all_dataset.reset('test', 0, transform=transform_test)
    testloader = torch.utils.data.DataLoader(all_dataset, batch_size=test_batch, shuffle=False, num_workers=4)
    rt_context.reset('test', transform=transform_test)
    context_loader = torch.utils.data.DataLoader(rt_context, batch_size=(num_domain-1)*NUM_CLASS*ctx_test, shuffle=False, num_workers=4, drop_last=False, worker_init_fn=worker_init_fn)
    # pdb.set_trace()
    for batch_idx, (inputs, targets) in enumerate(context_loader):
        context_img, context_label = inputs.to(device), targets.to(device)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            ys, _, _, _ = net(inputs, context_img, len(domains), ctx_test, ifsample)

            # ys = ys.mean(1)                     ## ## no ensemble
            ys = ys.view(ys.size()[0], num_preds, args.num_classes)

            y = torch.softmax(ys, -1).mean(1)
           
            cls_loss = criterion(y, targets)
            loss = cls_loss

            test_loss += loss.item()
            _, predicted = y.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            for i in range(num_preds):
                _, ac_predicted = ys[:, i].max(1)
                ac_correct[i] += ac_predicted.eq(targets).sum().item()

        log_string('TEST Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        writer.add_scalar('test_loss', test_loss/(batch_idx+1), epoch)
        writer.add_scalar('test_acc', 100.*correct/total, epoch)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        log_string('The best test Acc')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        # torch.save(state, os.path.join(LOG_DIR, 'ckpt.t7'))
        best_acc = acc
        return 0
    else:
        return 1


decay_ite = [0.6*max_ite]

if args.autodecay:
    for epoch in range(300):
        train(epoch)
        f = test(epoch)
        if f == 0:
            converge_count = 0
        else:
            converge_count += 1

        if converge_count == 20:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.2
            log_string('In epoch %d the LR is decay to %f' %(epoch, optimizer.param_groups[0]['lr']))
            converge_count = 0

        if optimizer.param_groups[0]['lr'] < 2e-6:
            exit()

else:
    if not iteration_training:
        for epoch in range(start_epoch, start_epoch+decay_inter[-1]+50):
            if epoch in decay_inter:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.1
                log_string('In epoch %d the LR is decay to %f' %(epoch, optimizer.param_groups[0]['lr']))
            train(epoch)
            if epoch % 5 == 0:
                _ = validation(epoch)
                _ = test(epoch)
    else:
        for epoch in range(max_ite):   
            if epoch in decay_ite:
                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = optimizer.param_groups[i]['lr']*0.1
                log_string('In iteration %d the LR is decay to %f' %(epoch, optimizer.param_groups[0]['lr']))
            train(epoch)
            if epoch % test_ite == 0:
                if args.dataset!='office':
                    _ = validation(epoch)
                _ = test(epoch)
