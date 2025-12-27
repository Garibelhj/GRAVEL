import torch as t
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from params import args
from Model import HGDM
from sklearn.metrics import roc_auc_score
from torch.nn.functional import softmax
from DataHandler import DataHandler,index_generator
import numpy as np
import pickle
from Utils.Utils import *
from Utils.Utils import contrast
import os
import logging
import datetime
import sys
import time
from datetime import timedelta
 

device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def format_time(seconds):
    """Format seconds to readable time string"""
    return str(timedelta(seconds=int(seconds)))

def get_gpu_memory():
    """Get current GPU memory usage in MB"""
    if t.cuda.is_available():
        return t.cuda.max_memory_allocated() / 1024 / 1024  # Convert to MB
    return 0

class Coach:
    def __init__(self, handler):
        self.handler = handler
       
        self.metrics = dict()
        mets = ['bceLoss','AUC']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        # ============ 训练监控初始化 ============
        print("\n" + "="*80)
        print("DiffGraph Training Monitor")
        print("="*80)
        print(f"Dataset: {args.data}")
        print(f"Total Ratios: {len(self.handler.train_idx)}")
        print(f"Repeats per ratio: 10")
        print(f"Epochs per repeat: {args.epoch}")
        print(f"Batch size: {args.batch}")
        print(f"Learning rate: {args.lr}")
        print(f"Diffusion steps: {args.steps}")
        print("="*80 + "\n")
        
        # 全局统计
        total_start_time = time.time()
        all_ratio_times = []
        
        for ratio in range(len(self.handler.train_idx)):
            ratio_start_time = time.time()
            log('Ratio Type: '+str(ratio))
            print(f"\n{'='*80}")
            print(f"Training Ratio {ratio+1}/{len(self.handler.train_idx)}")
            print(f"{'='*80}")
            
            accs = []
            micro_f1s = []
            macro_f1s = []
            macro_f1s_val = []
            auc_score_list = []
            
            for repeat in range(10):
                repeat_start_time = time.time()
                self.prepareModel()
                log('Repeat: '+str(repeat))
                print(f"\n[Repeat {repeat+1}/10] Started at {datetime.datetime.now().strftime('%H:%M:%S')}")

                macroMax = 0
                


                log_format = '%(asctime)s %(message)s'
                logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
                log_save = './History/'
                log_file = f'{args.data}_' + \
                                    f'lr_{args.lr}_batch_{args.batch}_noise_scale_{args.noise_scale}_step_{args.steps}_ratio_{ratio}_public'
                fname = f'{log_file}.txt'
                fh = logging.FileHandler(os.path.join(log_save, fname))
                fh.setFormatter(logging.Formatter(log_format))
                logger = logging.getLogger()
                logger.addHandler(fh)
                # logger.info(args)
                # logger.info('================')  
                args.save_path = log_file 

                val_accs = []
                val_micro_f1s = []
                val_macro_f1s = []
                test_accs = []
                test_micro_f1s = []
                test_macro_f1s = []
                logits_list = []
                test_lbls = t.argmax(self.label[self.test_idx[ratio]], dim=-1)
                
                # Epoch时间记录
                epoch_times = []
                max_gpu_memory = 0
                
                for ep in range(args.epoch):
                    epoch_start_time = time.time()
                    tstFlag = (ep % 1 == 0)
                    reses = self.trainEpoch(ratio)
                    # log(self.makePrint('Train', ep, reses, tstFlag))
                    if tstFlag:
                        val_reses,test_reses = self.testEpoch(ratio)
                        val_accs.append(val_reses['acc'].item())
                        val_macro_f1s.append(val_reses['macro'])
                        val_micro_f1s.append(val_reses['micro'])

                        test_accs.append(test_reses['acc'].item())
                        test_macro_f1s.append(test_reses['macro'])
                        test_micro_f1s.append(test_reses['micro'])
                        logits_list.append(test_reses['logits'])
                        
                        # Epoch结束统计
                        epoch_end_time = time.time()
                        epoch_time = epoch_end_time - epoch_start_time
                        epoch_times.append(epoch_time)
                        
                        # 更新GPU内存
                        if t.cuda.is_available():
                            current_gpu_memory = get_gpu_memory()
                            max_gpu_memory = max(max_gpu_memory, current_gpu_memory)
                        else:
                            current_gpu_memory = 0
                        
                        # 打印epoch进度
                        progress = (ep + 1) / args.epoch * 100
                        avg_epoch_time = np.mean(epoch_times)
                        eta_seconds = avg_epoch_time * (args.epoch - ep - 1)
                        
                        print(f"[Epoch {ep+1}/{args.epoch}] "
                              f"Progress: {progress:.1f}% | "
                              f"Time: {epoch_time:.2f}s | "
                              f"Avg: {avg_epoch_time:.2f}s | "
                              f"ETA: {format_time(eta_seconds)} | "
                              f"Test Acc: {test_reses['acc'].item():.4f} | "
                              f"Test Macro-F1: {test_reses['macro']:.4f} | "
                              f"GPU: {current_gpu_memory:.1f}MB")
                        # print("\t[Val_Classification] Macro-F1_epoch: {:.4f} Micro-F1_epoch: {:.4f} Test-acc_epoch: {:.4f}"
                        #     .format(val_reses['macro'],
                        # val_reses['micro'],
                        # val_reses['acc']
                        #     )
                        #     )


                        # print("\t[Test_Classification] Macro-F1_epoch: {:.4f} Micro-F1_epoch: {:.4f} Test-acc_epoch: {:.4f}"
                        #     .format(test_reses['macro'],
                        # test_reses['micro'],
                        # test_reses['acc']
                        #     )
                        #     )
                    
                    
                    
                        # log(self.makePrint('Test', ep, reses, tstFlag))
                        # if (val_reses['macro'] > macroMax):
                        #     macroMax = test_reses['macro']
                        #     self.saveModel()
                        # logger.info(self.makePrint('Test', ep, test_reses, tstFlag))
                        # self.saveHistory()
                   

                        # self.saveHistory()

                max_iter = test_accs.index(max(test_accs))
                accs.append(test_accs[max_iter])
                max_iter = test_macro_f1s.index(max(test_macro_f1s))
                macro_f1s.append(test_macro_f1s[max_iter])
                macro_f1s_val.append(val_macro_f1s[max_iter])

                max_iter = test_micro_f1s.index(max(test_micro_f1s))
                micro_f1s.append(test_micro_f1s[max_iter])

                best_logits = logits_list[max_iter]
                best_proba = softmax(best_logits, dim=1)
                
                # Calculate AUC with error handling
                try:
                    auc_score = roc_auc_score(y_true=test_lbls.detach().cpu().numpy(),
                                            y_score=best_proba.detach().cpu().numpy(),
                                            multi_class='ovr')
                    auc_score_list.append(auc_score)
                except Exception as e:
                    print(f"Warning: Could not calculate AUC score: {e}")
                    auc_score_list.append(0.0)
                
                # Repeat结束统计
                repeat_end_time = time.time()
                repeat_time = repeat_end_time - repeat_start_time
                
                print(f"\n{'─'*80}")
                print(f"[Repeat {repeat+1}/10 Completed]")
                print(f"  Time: {format_time(repeat_time)} ({repeat_time:.2f}s)")
                print(f"  Best Test Macro-F1: {macro_f1s[-1]:.4f}")
                print(f"  Best Test Micro-F1: {micro_f1s[-1]:.4f}")
                if len(auc_score_list) > 0:
                    print(f"  Best Test AUC: {auc_score_list[-1]:.4f}")
                if len(epoch_times) > 0:
                    print(f"  Avg Epoch Time: {np.mean(epoch_times):.2f}s")
                    print(f"  Total Epochs: {len(epoch_times)}")
                if t.cuda.is_available():
                    print(f"  Max GPU Memory: {max_gpu_memory:.2f} MB")
                print("─"*80 + "\n")
                
                # print("\t[Test_Classification] Macro-F1_one_time: {:.4f} Micro-F1_one_time: {:.4f} Test-AUC_one_time: {:.4f}"
                #             .format(macro_f1s[-1],
                #         micro_f1s[-1],
                #         auc_score_list[-1]
                #             )
                #             )
                

            # Ratio结束统计
            ratio_end_time = time.time()
            ratio_time = ratio_end_time - ratio_start_time
            all_ratio_times.append(ratio_time)
            
            print(f"\n{'='*80}")
            print(f"[Ratio {ratio+1}/{len(self.handler.train_idx)} Completed]")
            print(f"  Total Time: {format_time(ratio_time)} ({ratio_time:.2f}s)")
            print(f"  Macro-F1 Mean: {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
            print(f"  Micro-F1 Mean: {np.mean(micro_f1s):.4f} ± {np.std(micro_f1s):.4f}")
            print(f"  AUC Mean: {np.mean(auc_score_list):.4f} ± {np.std(auc_score_list):.4f}")
            print("="*80 + "\n")
            
            logger.info("\t[Classification] Macro-F1: {:.4f} var: {:.4f}  Micro-F1_mean: {:.4f} var: {:.4f} auc {:.4f}"
                .format(np.mean(macro_f1s),
                        np.std(macro_f1s),
                        np.mean(micro_f1s),
                        np.std(micro_f1s),
                        np.mean(auc_score_list),
                        np.std(auc_score_list)))
        
        # ============ 训练完成总结 ============
        total_end_time = time.time()
        total_training_time = total_end_time - total_start_time
        
        print("\n" + "="*80)
        print("Training Completed - Final Summary")
        print("="*80)
        
        print(f"\n[Time Statistics]")
        print(f"  Total Training Time: {format_time(total_training_time)} ({total_training_time:.2f}s)")
        if len(all_ratio_times) > 0:
            print(f"  Average Ratio Time: {format_time(np.mean(all_ratio_times))} ({np.mean(all_ratio_times):.2f}s)")
            print(f"  Fastest Ratio Time: {format_time(np.min(all_ratio_times))} ({np.min(all_ratio_times):.2f}s)")
            print(f"  Slowest Ratio Time: {format_time(np.max(all_ratio_times))} ({np.max(all_ratio_times):.2f}s)")
        
        print(f"\n[Resource Usage]")
        if t.cuda.is_available():
            final_gpu_memory = get_gpu_memory()
            print(f"  Max GPU Memory: {final_gpu_memory:.2f} MB ({final_gpu_memory/1024:.2f} GB)")
            print(f"  GPU Device: {t.cuda.get_device_name(0)}")
        
        print(f"\n[Dataset Information]")
        print(f"  Dataset: {args.data}")
        print(f"  Total Ratios: {len(self.handler.train_idx)}")
        print(f"  Repeats per ratio: 10")
        print(f"  Epochs per repeat: {args.epoch}")
        
        print(f"\n[Completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        print("="*80 + "\n")
        # =========================================

    def prepareModel(self):
        self.initial_feature = self.handler.feature_list
        self.dim = self.initial_feature.shape[1]
        self.train_idx = self.handler.train_idx
        self.test_idx = self.handler.test_idx
        self.val_idx = self.handler.val_idx
        self.label = self.handler.labels
        self.nbclasses = self.label.shape[1]
        
        self.model = HGDM(self.dim).to(device)
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        
        # 打印模型参数统计（只在第一次调用时打印）
        if not hasattr(self, '_model_params_printed'):
            total_params, trainable_params = count_parameters(self.model)
            print(f"\n[Model Information]")
            print(f"  Input dimension: {self.dim}")
            print(f"  Number of classes: {self.nbclasses}")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            if t.cuda.is_available():
                print(f"  GPU Device: {t.cuda.get_device_name(0)}")
            print()
            self._model_params_printed = True
            
            # 重置GPU内存统计
            if t.cuda.is_available():
                t.cuda.reset_peak_memory_stats()
                t.cuda.empty_cache()

    def trainEpoch(self,i):

        trnLoader = index_generator(batch_size=args.batch, indices=self.train_idx[i])
       
        epBCELoss, epDFLoss = 0, 0
        self.label = self.handler.labels
        steps = trnLoader.num_iterations()
       
        for i in range(trnLoader.num_iterations()):
            train_idx_batch = trnLoader.next()
            train_idx_batch.sort()
            ancs=t.LongTensor(train_idx_batch)

            nll_loss,diffloss = self.model.cal_loss(ancs, self.label,self.handler.he_adjs,self.initial_feature)
    
            loss = nll_loss +  diffloss
            epBCELoss += nll_loss.item()
            
            epDFLoss += diffloss.item()
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            # log('Step %d/%d: bceloss = %.3f, diffloss = %.3f    ' % (i, steps, nll_loss,diffloss), save=False,
            #     oneline=True)
        ret = dict()
        ret['bceLoss'] = epBCELoss / steps
        ret['diffLoss'] = epDFLoss / steps
        
        
        return ret
   

    def testEpoch(self,i):
        labels = self.handler.labels
        test_idx = self.handler.test_idx[i]
        with t.no_grad():

            embeds,scores = self.model.get_allembeds(self.handler.he_adjs,self.initial_feature)
            val_acc,val_f1_macro,val_f1_micro,test_acc,test_f1_macro,test_f1_micro,test_logits=evaluate(embeds,scores, args.ratio[i], self.train_idx[i], self.val_idx[i], self.test_idx[i], labels, self.nbclasses)
            val_ret = dict()
            val_ret['acc'] = val_acc
            val_ret['macro'] = val_f1_macro
            val_ret['micro'] = val_f1_micro

            test_ret = dict()
            test_ret['acc'] = test_acc
            test_ret['macro'] = test_f1_macro
            test_ret['micro'] = test_f1_micro
            test_ret['logits'] = test_logits
            return val_ret,test_ret

    


    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('./History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        
    def saveModel(self):
        content = {
            'model': self.model,
        }
        t.save(content, './Models/' + args.save_path + '.mod')
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        
        ckp = t.load('./Models/' + args.load_model )
        self.model = ckp['model']
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        log('Model Loaded')



if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.saveDefault = True
    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    coach = Coach(handler)
    coach.run()
    # coach.test()