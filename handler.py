
from IPython.display import display
from models import *
from config import *
from src.utils import *

def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, 'StemGNN.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)


def load_model(model_dir, epoch=None):
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, 'StemGNN.pt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model


def inference(model, dataloader, device, node_cnt, window_size, horizon):
    forecast_set = []
    target_set = []
    model.eval()
    with torch.no_grad():
        for i, (inputs, target) in enumerate(dataloader):
            inputs = inputs.to(device)
            target = target.to(device)
            step = 0
            forecast_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float64)
            while step < horizon:
                forecast_result, a = model(inputs)
                len_model_output = forecast_result.size()[1] #node_cnt 
                if len_model_output == 0:
                    raise Exception('Get blank inference result')
                inputs[:, :window_size - len_model_output, :] = inputs[:, len_model_output:window_size, :].clone()
                inputs[:, window_size - len_model_output:, :] = forecast_result.clone()
                forecast_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                    forecast_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()
                
                step += min(horizon - step, len_model_output)
            forecast_set.append(forecast_steps)
            target_set.append(target.detach().cpu().numpy())
    return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0)

def validate(model, dataloader, device, normalize_method, statistic,
             node_cnt, window_size, horizon):
                
    forecast_norm, target_norm = inference(model, dataloader, device,
                                           node_cnt, window_size, horizon)
    if normalize_method and statistic:
        forecast = de_normalized(forecast_norm, normalize_method, statistic)
        target = de_normalized(target_norm, normalize_method, statistic)
    else:
        forecast, target = forecast_norm, target_norm
    
    score = evaluate(np.array(target[0][0]), np.array(forecast[0][0]))
    score_by_node = evaluate(target, forecast, by_node=True)
    score_norm = evaluate(target_norm, forecast_norm)
    
    return dict(mae=score[1], mae_node=score_by_node[1], mape=score[0], mape_node=score_by_node[0],
                rmse=score[2], rmse_node=score_by_node[2],forecast=forecast, target=target)

def train(data, args,result_file):
    node_cnt = data.shape[1]
    model = Model(node_cnt, 2, args.window_size, args.multi_layer, horizon=args.horizon)
    model.to(args.device)
    if len(data) == 0:
        raise Exception('Cannot organize enough training data')

    if args.norm_method == 'z_score':
        train_mean = np.mean(data, axis=0)
        train_std = np.std(data, axis=0)
        normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
    elif args.norm_method == 'min_max':
        train_min = np.min(data, axis=0)
        train_max = np.max(data, axis=0)
        normalize_statistic = {"min": train_min.tolist(), "max": train_max.tolist()}
    else:
        normalize_statistic = None
    if normalize_statistic is not None:
        with open(os.path.join(result_file, 'norm_stat.json'), 'w') as f:
            json.dump(normalize_statistic, f)
    
    if args.optimizer == 'RMSProp':
        my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.lr, eps=1e-08)
    else:
        my_optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)

    train_set = ForecastDataset(data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                         num_workers=0)
    forecast_loss = nn.MSELoss(reduction='mean').to(args.device)

    best_validate_mae = np.inf
    validate_score_non_decrease_count = 0
    performance_metrics = {}

    slicing_valid_sample = list(set([random.randrange(1,len(train_set)-1) for i in range(int(len(train_set)*args.valid_ratio))]))
    slicing_train,slicing_valid,slicing_test = [],[],[] 

    for i,(x,y) in enumerate(train_set) :
        if i in [len(train_set)-1] :
            slicing_test.append((x,y))
        elif i in (slicing_valid_sample) :
            slicing_valid.append((x,y))
        else :
            slicing_train.append((x,y))
            
    train_loader = torch_data.DataLoader(slicing_train, batch_size=args.batch_size, drop_last=False, shuffle=True, num_workers=0)
    valid_loader = torch_data.DataLoader(slicing_valid, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        for i, (inputs, target) in enumerate(train_loader):
            
            inputs = inputs.to(args.device)
            target = target.to(args.device)
            model.zero_grad()
            forecast, attention_tmp = model(inputs)
            loss = forecast_loss(forecast, target)
            cnt += 1
            loss.backward()
            my_optim.step()
            loss_total += float(loss)
            
        if (epoch + 1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
        if (epoch + 1) % args.validate_freq == 0:
            print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f}'.format(epoch+1, (time.time() - epoch_start_time), loss_total / cnt))

        is_best_for_now = False
                
        pm = validate(model, valid_loader, args.device, args.norm_method, normalize_statistic,
                     node_cnt, args.window_size, args.horizon)
        
        if best_validate_mae > pm['mae']:
            best_mae,best_mape,best_rmse = pm['mae'],pm['mape'],pm['rmse']
            is_best_for_now = True
            validate_score_non_decrease_count = 0
        else:
            validate_score_non_decrease_count += 1
            
        if is_best_for_now:
            save_model(model, result_file, epoch)
        if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
            print('Epoch:{}'.format(epoch),end=' | ')
            break
    
    print('\n'+'#'*20)
    print('Best validate performance:')
    print(f'MAE:{round(best_mae, 3)} | MAPE:{round(best_mape, 3)} | RMSE:{round(best_rmse, 3)}')
    print('#'*20+'\n')

    return 0


def test(data, test_set, args, result_test_file):
    
    node_cnt = data.shape[1]
    with open(os.path.join(result_test_file, 'norm_stat.json'),'r') as f:
        normalize_statistic = json.load(f)

    model = load_model(result_test_file)
    test_loader = torch_data.DataLoader(test_set, batch_size=args.batch_size, drop_last=False,
                                        shuffle=False, num_workers=0)
    
    performance_metrics = validate(model, test_loader, args.device, args.norm_method, normalize_statistic,
                  node_cnt, args.window_size, args.horizon, station_num=args.station_index)
    
    return performance_metrics

def stemgnn (args) :

    result_test_file = os.path.join('result', args.dataset)
    if not os.path.exists(result_test_file):
        os.makedirs(result_test_file)
    data_file = os.path.join('dataset', args.dataset + '.csv')
    data = pd.read_csv(data_file)
    data = data.values

    #parser.add_argument('--early_stop_step', type=int, default=500)
    #parser.add_argument('--exponential_decay_step', type=int, default=100)
    args.exponential_decay_step = args.epoch//2
    args.early_stop_step = args.epoch

    try:
        inference = train(data, args, result_test_file)
        #performance_metrics = test(data, inference, args, result_test_file)
    except KeyboardInterrupt:
        print('\n'+'#'*20)
        print('Keyboard Interrupt')
        print('#'*20+'\n')


def main (args) :
  passenger_flow = stemgnn(args)
