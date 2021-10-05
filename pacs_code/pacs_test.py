from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import torch.backends.cudnn as cudnn
from torchvision import models
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
import sys

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='learning rate')
parser.add_argument('--sparse', default=0, type=float, help='L1 panelty')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='res18_cartoon', help='Log dir [default: log]')
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
parser.add_argument('--zerozd', default=0, type=int, help='hierarchical model')

args = parser.parse_args()
BATCH_SIZE = args.batch_size
OPTIMIZER = args.optimizer
gpu_index = args.gpu
backbone = args.net
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
zerozd = args.zerozd
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

LOG_DIR = os.path.join('logs', 'test_' + args.log_dir)
# args.log_dir = LOG_DIR
# pdb.set_trace()

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


# Data
print('==> Preparing data..')

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

# Model
resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)

class classifier_generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(classifier_generator, self).__init__()
        self.shared_net = nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.ReLU()
                        )
        self.shared_mu = nn.Linear(hidden_size, output_size)
        self.shared_sigma = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        z = self.shared_net(x)
        return self.shared_mu(z), self.shared_sigma(z)

class net0(nn.Module):
    def __init__(self, num_class, num_source_domains, backbone, dinit, agg_model='concat', agg_method='ensemble', dsgrad=True, 
                zdlayers=4, dom_sta='both', prior_test=False, hierar=0, adap_samples=1, multi_train_samp=False, sharemodel=True, mc_times=10, with_bias=True):
        super(net0, self).__init__()
        self.num_class = num_class
        self.with_bias = with_bias
        self.num_d = num_source_domains
        self.mc_times = mc_times
        self.zdlayers = zdlayers
        self.dinit = dinit
        self.agg_model = agg_model
        self.agg_method = agg_method
        self.prior_test = prior_test
        self.sharemodel = sharemodel
        self.dom_sta = dom_sta
        self.hierar = hierar
        self.dsgrad = dsgrad
        self.test_adap_samples = adap_samples
        if multi_train_samp:
            self.train_adap_samples = adap_samples
        else:
            self.train_adap_samples = '1'

        # backbone
        if backbone=='res18':
            resnet = resnet18
            self.feature_dim = 512

            if zdlayers==4:
                len_zd = (resnet.layer1[-1].conv2.out_channels + resnet.layer2[-1].conv2.out_channels + resnet.layer3[-1].conv2.out_channels + resnet.layer4[-1].conv2.out_channels) * (2**int((dom_sta=='both')))
            elif zdlayers==3:
                len_zd = (resnet.layer1[-1].conv2.out_channels + resnet.layer2[-1].conv2.out_channels + resnet.layer3[-1].conv2.out_channels) * (2**int((dom_sta=='both')))
            elif zdlayers==1:
                len_zd = (resnet.layer4[-1].conv2.out_channels) * (2**int((dom_sta=='both')))

        elif backbone=='res50':
            resnet = resnet50
            self.feature_dim = 2048

            if zdlayers==4:
                len_zd = (resnet.layer1[-1].conv3.out_channels + resnet.layer2[-1].conv3.out_channels + resnet.layer3[-1].conv3.out_channels + resnet.layer4[-1].conv3.out_channels) * (2**int((dom_sta=='both')))
            elif zdlayers==3:
                len_zd = (resnet.layer1[-1].conv3.out_channels + resnet.layer2[-1].conv3.out_channels + resnet.layer3[-1].conv3.out_channels) * (2**int((dom_sta=='both')))
            elif zdlayers==1:
                len_zd = (resnet.layer4[-1].conv3.out_channels) * (2**int((dom_sta=='both')))

        self.layer0 = nn.Sequential(
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu,
                    resnet.maxpool
                    )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.pool = resnet.avgpool

        # aggregation
        self.domain_layer = nn.Sequential(
                            nn.Linear(len_zd, self.feature_dim),
                            nn.ReLU(),
                            nn.Linear(self.feature_dim, self.feature_dim),
                            )
        self.agg_layer = nn.Sequential(
                        nn.Linear(self.feature_dim*2, self.feature_dim),
                        nn.ReLU(),
                        nn.Linear(self.feature_dim, self.feature_dim),
                        )

        self.shared_net = classifier_generator(self.feature_dim, self.feature_dim, self.feature_dim)
        if hierar==2:
            self.shared_net1 = classifier_generator(self.feature_dim, self.feature_dim, self.feature_dim)

        # context info
        self.context = []

    def forward(self, x, ctx, domain_id=0, num_ctx=20, ifsample=True):
        # pdb.set_trace()
        self.num_ctx = num_ctx
        self.domain_id = domain_id

        z0 = self.layer0(x)
        z1 = self.layer1(z0)
        z2 = self.layer2(z1)
        z3 = self.layer3(z2)
        z4 = self.layer4(z3)
        z = self.pool(z4)

        ctx = self.layer0(ctx)
        ctx = self.layer1(ctx)
        ctx = self.layer2(ctx)
        ctx = self.layer3(ctx)
        ctx = self.layer4(ctx)
        ctx = self.pool(ctx)
        # pdb.set_trace()

        ctx = ctx.view(3, self.num_class, num_ctx, self.feature_dim)

        ctx = ctx.mean(2)

        if not self.dsgrad:
            ctx = ctx.detach()

        # pdb.set_trace()
        z1 = z1.mean(-1).mean(-1)
        z2 = z2.mean(-1).mean(-1)
        z3 = z3.mean(-1).mean(-1)
        z4 = z4.mean(-1).mean(-1)

        if self.zdlayers==4:
            zd = torch.cat([z1, z2, z3, z4], 1)
        elif self.zdlayers==3:
            zd = torch.cat([z1, z2, z3], 1)
        elif self.zdlayers==1:
            zd = z4
        self.z = z.squeeze()
        # print(self.z)
        zd = zd.squeeze()
        #zd = torch.zeros(zd.size()).cuda()
        # pdb.set_trace()
        # yc, common_kl = self.com_classifier(self.z, self.mc_times, ifsample)
        
        if self.training:
            self.ws, pws, KLD, hkld = self.adaptive_weight_training(zd, self.z, ctx, domain_id) # 128*10*14*512
            # pdb.set_trace()
            pws = pws.view(self.feature_dim, self.mc_times*self.num_class) # 512*(10*7)
            yps = torch.mm(self.z, pws) # 128*512  512*(10*7) => 128*(10*7)
            yps = yps.view(yps.size()[0], self.mc_times, self.num_class)
            # pdb.set_trace()
        else:
            self.ws, KLD, hkld = self.target_adaptive_weight(zd, self.z, ctx, ifsample)
                # pdb.set_trace()
            yps = 0 
            hkld = 0

        self.ws = self.ws.view(z.size()[0], -1, self.feature_dim) # 128*(10*2*7)*512
        ys = torch.bmm(self.ws, self.z.unsqueeze(-1)) # 128*(10*2*7)*512   128*512*1 => 128*(10*2*7)*1
        ys = ys.squeeze() # 128*(10*2*7)

        ys = ys.view(ys.size()[0], self.mc_times**int(ifsample), -1) #128*10*2*7  common adapt in test
        # ys = ys.view(ys.size()[0], self.mc_times, -1) #128*10*2*7   test no adapt

        # return yc, ys, yps, common_kl, KLD
        return ys, yps, KLD, hkld

    def adaptive_weight_training(self, zd, zt, c, domain_id, add_kl=False):

        self.context = []
        for i in range(c.size()[0]):
            self.context.append(c[i])

        current_target_domain = self.context.pop(domain_id)

        num_sd = 1
        current_source_domains = torch.cat(self.context, 0).view(len(self.context), self.num_class, self.feature_dim).mean(0) # 7*512

        p_mu, p_v = self.shared_net(current_target_domain) # 7*512
            
        self.context.insert(domain_id, current_target_domain)

        # posterior
        if self.zdlayers==1:
            zd = zd
        else:
            zd = self.domain_layer(zd)  # 128*512

        if self.train_adap_samples == 'all':
            # pdb.set_trace()
            bs = zd.size()[0]
            zd = zd.mean(0, True).repeat(bs, 1)
        elif self.train_adap_samples != '1':
            # pdb.set_trace()
            # zd = zd.view(-1, int(self.train_adap_samples), self.feature_dim).mean(1, True)
            # zd = zd.repeat(1, int(self.train_adap_samples), 1).view(-1, self.feature_dim)
            multi_feat = zd[:int(self.train_adap_samples)-1]
            # pdb.set_trace()
            multi_feat = multi_feat.sum(0, True)
            multi_feat = multi_feat.repeat(zd.size()[0], 1)
            zd = (zd + multi_feat) / int(self.train_adap_samples)

        if self.hierar==0:
            ws = current_source_domains
            # H_kld = torch.zeros(self.num_class, self.mc_times, self.feature_dim).cuda()
            H_kld = torch.zeros(1).cuda()
        elif self.hierar==2:
            ws_mu, ws_v = self.shared_net1(current_source_domains) # (2*7)*512
            # pdb.set_trace()
            ws_sigma = f.softplus(ws_v, beta=1, threshold=20)
            ws_mu_samp = ws_mu.unsqueeze(1).repeat(1, self.mc_times, 1) # (2*7)*10*512
            ws_sigma_samp = ws_sigma.unsqueeze(1).repeat(1, self.mc_times, 1) # (2*7)*10*512
            # pdb.set_trace()
            eps_ws = ws_mu_samp.new(ws_mu_samp.size()).normal_()
            ws = ws_mu_samp + 1 * ws_sigma_samp * eps_ws # (2*7)*10*512
            # ws = ws_mu_samp

            H_kld = torch.zeros(1).cuda()

        # pdb.set_trace()
        zd = zd.view(zd.size()[0], 1, 1, zd.size()[1]).repeat(1, num_sd*self.num_class, self.mc_times**int(self.hierar!=0), 1) # 128*14*10*512
        if self.agg_method!='att':
            Ds = ws.view(1, num_sd*self.num_class, self.mc_times**int(self.hierar!=0), self.feature_dim).repeat(zd.size()[0], 1, 1, 1) # 128*14*10*512
        else:
            Ds = ws.view(zd.size()[0], num_sd*self.num_class, self.mc_times**int(self.hierar!=0), self.feature_dim)
        q = self.agg_layer(torch.cat([zd, Ds], -1).view(zd.size()[0]*zd.size()[1]*zd.size()[2], zd.size()[-1]+Ds.size()[-1]))
        if self.hierar==3:
            q_mu, q_v = self.shared_net2(q) # (128*14*10)*512
        else:
            q_mu, q_v = self.shared_net(q) # (128*14*10)*512
        q_mu = q_mu.view(zd.size()[0], num_sd*self.num_class, self.mc_times**int(self.hierar!=0), self.feature_dim).mean(2) # 128*14*10*512
        q_v = q_v.view(zd.size()[0], num_sd*self.num_class, self.mc_times**int(self.hierar!=0), self.feature_dim).mean(2) # 128*14*10*512

        # KL divergence
        p_sigma = f.softplus(p_v, beta=1, threshold=20)
        q_sigma = f.softplus(q_v, beta=1, threshold=20)

        p_mu_re = p_mu.view(1, 1, self.num_class, self.feature_dim).repeat(zd.size()[0], num_sd, 1, 1) #128*2*7*512
        p_sigma_re = p_sigma.view(1, 1, self.num_class, self.feature_dim).repeat(zd.size()[0], num_sd, 1, 1)
        p_mu_re = p_mu_re.view(zd.size()[0], num_sd*self.num_class, self.feature_dim) #128*(2*7)*512
        p_sigma_re = p_sigma_re.view(zd.size()[0], num_sd*self.num_class, self.feature_dim)

        KLD = self.kl_divergence(q_mu, q_sigma, p_mu_re, p_sigma_re)
        KLD = KLD.sum()

        # sampling
        q_mu_samp = q_mu.unsqueeze(1).repeat(1, self.mc_times, 1, 1) # 128*10*(2*7)*512
        q_sigma_samp = q_sigma.unsqueeze(1).repeat(1, self.mc_times, 1, 1) # 128*10*(2*7)*512
        eps_q = q_mu_samp.new(q_mu_samp.size()).normal_()
        qw = q_mu_samp + 1 * q_sigma_samp * eps_q

        # pdb.set_trace()
        p_mu_samp = p_mu.transpose(0,1).unsqueeze(1).repeat(1, self.mc_times, 1) # 512*10*7
        p_sigma_samp = p_sigma.transpose(1,0).unsqueeze(1).repeat(1, self.mc_times, 1) # 512*10*7
        eps_p = p_mu_samp.new(p_mu_samp.size()).normal_()
        pw = p_mu_samp + 1 * p_sigma_samp * eps_p

        print(q_mu.mean(), q_sigma.mean())
        print(p_mu.mean(), p_sigma.mean())

        return qw, pw, KLD, H_kld

    def target_adaptive_weight(self, zd, zt, c, ifsample):
        # # pdb.set_trace()
        self.context = []
        for i in range(c.size()[0]):
            self.context.append(c[i])

        # pdb.set_trace()
        if self.domain_id < 3:
            current_target_domain = self.context.pop(self.domain_id)

        # pdb.set_trace()

        num_sd = 1
        current_source_domains = torch.cat(self.context, 0).view(len(self.context), self.num_class, self.feature_dim).mean(0) # 7*512

        if self.domain_id < 3:
            self.context.insert(self.domain_id, current_target_domain)

        # posterior
        if self.hierar==0:
            ws = current_source_domains
        elif self.hierar==2:
            ws_mu, ws_v = self.shared_net1(current_source_domains) # (3*7)*512
            # pdb.set_trace()
            ws_sigma = f.softplus(ws_v, beta=1, threshold=20)
            ws_mu_samp = ws_mu.unsqueeze(1).repeat(1, self.mc_times, 1) # (3*7)*10*512
            ws_sigma_samp = ws_sigma.unsqueeze(1).repeat(1, self.mc_times, 1) # (3*7)*10*512
            eps_ws = ws_mu_samp.new(ws_mu_samp.size()).normal_()
            # ws = ws_mu_samp + 1 * ws_sigma_samp * eps_ws # (3*7)*10*512
            ws = ws_mu_samp

        zd = self.domain_layer(zd)  # 128*512

        if self.test_adap_samples == 'all':
            # pdb.set_trace()
            bs = zd.size()[0]
            zd = zd.mean(0, True).repeat(bs, 1)
        elif self.test_adap_samples != '1':
            # pdb.set_trace()
            # zd = zd.view(-1, int(self.test_adap_samples), self.feature_dim).mean(1, True)
            # zd = zd.repeat(1, int(self.test_adap_samples), 1).view(-1, self.feature_dim)
            multi_feat = zd[:int(self.test_adap_samples)-1]
            # pdb.set_trace()
            multi_feat = multi_feat.sum(0, True)
            multi_feat = multi_feat.repeat(zd.size()[0], 1)
            zd = (zd + multi_feat) / int(self.test_adap_samples)

        # pdb.set_trace()
        zd = zd.view(zd.size()[0], 1, 1, zd.size()[1]).repeat(1, num_sd*self.num_class, self.mc_times**int(self.hierar!=0), 1) # 128*14*10*512
        if self.agg_method!='att':
            Ds = ws.view(1, num_sd*self.num_class, self.mc_times**int(self.hierar!=0), self.feature_dim).repeat(zd.size()[0], 1, 1, 1) # 128*14*10*512
        else:
            Ds = ws.view(zd.size()[0], num_sd*self.num_class, self.mc_times**int(self.hierar!=0), self.feature_dim)
        # zd = zd.view(zd.size()[0], 1, 1, zd.size()[1]).repeat(1,current_source_domains.size()[0], self.mc_times**int(self.hierar!=0), 1) # 128*21*10*512
        # Ds = ws.view(1, num_sd*self.num_class, self.mc_times**int(self.hierar!=0), self.feature_dim).repeat(zd.size()[0], 1, 1, 1) # 128*21*10*512
        q = self.agg_layer(torch.cat([zd, Ds], -1).view(zd.size()[0]*zd.size()[1]*zd.size()[2], zd.size()[-1]+Ds.size()[-1]))
        if self.hierar==3:
            q_mu, q_v = self.shared_net2(q) # (128*21*10)*512
        else:
            q_mu, q_v = self.shared_net(q) # 128*21*10*512
        q_mu = q_mu.view(zd.size()[0], num_sd*self.num_class, self.mc_times**int(self.hierar!=0), self.feature_dim).mean(2)
        q_v = q_v.view(zd.size()[0], num_sd*self.num_class, self.mc_times**int(self.hierar!=0), self.feature_dim).mean(2)
        q_sigma = f.softplus(q_v, beta=1, threshold=20)

        KLD = 0
        H_kld = 0

        # sampling
        if ifsample:
            q_mu_samp = q_mu.unsqueeze(2).repeat(1, 1, self.mc_times, 1)
            q_sigma_samp = q_sigma.unsqueeze(2).repeat(1, 1, self.mc_times, 1)
            eps_q = q_mu_samp.new(q_mu_samp.size()).normal_()
            qw = q_mu_samp + 1 * q_sigma_samp * eps_q
        else:
            qw = q_mu

        return qw, KLD, H_kld

    def kl_divergence(self, mu_q, sigma_q, mu_p, sigma_p):

        var_q = sigma_q**2 + 1e-6
        var_p = sigma_p**2 + 1e-6

        component1 = torch.log(var_p) - torch.log(var_q)
        component2 = var_q / var_p
        component3 = (mu_p - mu_q).pow(2)/ var_p

        KLD = 0.5 * torch.sum((component1 -1 +component2 +component3),1)
        return KLD

print('==> Building model..')

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

net = net0(args.num_classes, len(domains), backbone, Dinit, agg_model, agg_method, zerozd, zdlayers, dom_sta, prior_test, hierar, num_adp, multi_train_sam, sharemodel)

net.apply(inplace_relu)

log_string(str(net.extra_repr))

pc = get_parameter_number(net)
log_string('Total: %.4fM, Trainable: %.4fM' %(pc['Total']/float(1e6), pc['Trainable']/float(1e6)))

net = net.to(device)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)

# if isinstance(net,torch.nn.DataParallel):
#     net = net.module
# net.train()
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# pdb.set_trace()

# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('logs/' + args.log_dir + '/ckpt.t7')
net.load_state_dict(checkpoint['net'])
    # best_acc = checkpoint['acc']
    # start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

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

            ys, _, _, _ = net(inputs, context_img, len(domains), ctx_test, ifsample)
            
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
            log_string('Epoch: %d | TEST Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        log_string('The best test Acc')
        best_acc = acc
        return 0
    else:
        return 1


_ = test(0)
