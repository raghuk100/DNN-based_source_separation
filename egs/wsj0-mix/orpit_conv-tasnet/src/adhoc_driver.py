import os
import time
import numpy as np
import torch
import torch.nn as nn

from utils.utils import draw_loss_curve
from utils.utils_audio import write_wav
from driver import TrainerBase, TesterBase

# TODO: torch.split for readility

class ORPITTrainer(TrainerBase):
    def __init__(self, model, loader, pit_criterion, optimizer, args):
        super().__init__(model, loader, pit_criterion, optimizer, args)
    
    def _reset(self, args):
        self.sr = args.sr
        self.n_sources = args.n_sources
        self.max_norm = args.max_norm
        
        self.model_dir = args.model_dir
        self.loss_dir = args.loss_dir
        self.sample_dir = args.sample_dir
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.loss_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        
        self.epochs = args.epochs
        self.train_loss = torch.empty(self.epochs)
        
        self.use_cuda = args.use_cuda
        
        if args.continue_from:
            package = torch.load(args.continue_from, map_location=lambda storage, loc: storage)

            self.start_epoch = package['epoch']
            self.train_loss[:self.start_epoch] = package['train_loss'][:self.start_epoch]
            
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(package['state_dict'])
            else:
                self.model.load_state_dict(package['state_dict'])
            
            self.optimizer.load_state_dict(package['optim_dict'])
        else:
            # TODO: redundant? last.pth never exists
            model_path = os.path.join(self.model_dir, "last.pth")
            
            if os.path.exists(model_path):
                if args.overwrite:
                    print("Overwrite models.")
                else:
                    raise ValueError("{} already exists. If you continue to run, set --overwrite to be True.".format(model_path))
            
            self.start_epoch = 0
    
    def run(self):
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            train_loss = self.run_one_epoch(epoch)
            end = time.time()
            
            print("[Epoch {}/{}] loss (train): {:.5f}, {:.3f} [sec]".format(epoch + 1, self.epochs, train_loss, end - start), flush=True)
            
            self.train_loss[epoch] = train_loss
            
            model_path = os.path.join(self.model_dir, "last.pth")
            self.save_model(epoch, model_path)
            
            save_path = os.path.join(self.loss_dir, "loss.png")
            draw_loss_curve(train_loss=self.train_loss[:epoch+1], save_path=save_path)
    
    def run_one_epoch(self, epoch):
        """
        Training
        """
        train_loss = self.run_one_epoch_train(epoch)
        _ = self.run_one_epoch_eval(epoch)

        return train_loss
    
    def run_one_epoch_eval(self, epoch):
        """
        Validation
        """
        self.model.eval()

        n_sources_count = {}
        
        with torch.no_grad():
            for idx, (mixture, sources, segment_IDs) in enumerate(self.valid_loader):
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                
                sources, n_sources = nn.utils.rnn.pad_packed_sequence(sources, batch_first=True)
                n_sources = n_sources.tolist()
                
                output_one_and_rest = self.model(mixture)
                output_one = output_one_and_rest[:, 0:1]
                output_rest = output_one_and_rest[:, 1:]
                output = output_one

                for source_idx in range(1, n_sources[0] - 1):
                    output_one_and_rest = self.model(output_rest)
                    output_one = output_one_and_rest[:, 0:1]
                    output_rest = output_one_and_rest[:, 1:]
                    output = torch.cat([output, output_one], dim=1)
                
                output = torch.cat([output, output_rest], dim=1)

                if not n_sources[0] in n_sources_count.keys():
                    n_sources_count[n_sources[0]] = 0
                
                if n_sources_count[n_sources[0]] < 5:
                    mixture = mixture[0].squeeze(dim=0).detach().cpu().numpy()
                    estimated_sources = output[0].detach().cpu().numpy()
                    
                    save_dir = os.path.join(self.sample_dir, segment_IDs[0])
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "mixture.wav")
                    norm = np.abs(mixture).max()
                    mixture = mixture / norm
                    write_wav(save_path, signal=mixture, sr=self.sr)
                    
                    for source_idx, estimated_source in enumerate(estimated_sources):
                        save_path = os.path.join(save_dir, "epoch{}-{}.wav".format(epoch + 1, source_idx + 1))
                        norm = np.abs(estimated_source).max()
                        estimated_source = estimated_source / norm
                        write_wav(save_path, signal=estimated_source, sr=self.sr)
                
                n_sources_count[n_sources[0]] += 1
        
        return -1
     
    def save_model(self, epoch, model_path='./tmp.pth'):
        if isinstance(self.model, nn.DataParallel):
            package = self.model.module.get_package()
            package['state_dict'] = self.model.module.state_dict()
        else:
            package = self.model.get_package()
            package['state_dict'] = self.model.state_dict()
            
        package['optim_dict'] = self.optimizer.state_dict()
        
        package['epoch'] = epoch + 1
        package['train_loss'] = self.train_loss
        
        torch.save(package, model_path)

class Tester(TesterBase):
    def __init__(self, model, loader, pit_criterion, args):
        super().__init__(self, model, loader, pit_criterion, args)

class AdhocTrainer(ORPITTrainer):
    def __init__(self, model, loader, pit_criterion, optimizer, args):
        super().__init__(model, loader, pit_criterion, optimizer, args)

class FinetuneTrainer(TrainerBase):
    def __init__(self, model, loader, pit_criterion, optimizer, args):
        super().__init__(model, loader, pit_criterion, optimizer, args)

    def _reset(self, args):
        self.sr = args.sr
        self.n_sources = args.n_sources
        self.max_norm = args.max_norm
        
        self.model_dir = args.model_dir
        self.loss_dir = args.loss_dir
        self.sample_dir = args.sample_dir
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.loss_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        
        self.epochs = args.epochs
        self.train_loss = torch.empty(self.epochs)
        self.valid_loss = torch.empty(self.epochs)
        
        self.use_cuda = args.use_cuda
        
        # Continue from
        package = torch.load(args.continue_from, map_location=lambda storage, loc: storage)

        continue_from_finetune = package.get('is_finetune') or False

        if continue_from_finetune:
            self.start_epoch = package['epoch']
            self.train_loss[:self.start_epoch] = package['train_loss'][:self.start_epoch]
            self.valid_loss[:self.start_epoch] = package['valid_loss'][:self.start_epoch]

            self.best_loss = package['best_loss']
        else:
            model_path = os.path.join(self.model_dir, "finetuned_best.pth")
            
            if os.path.exists(model_path):
                if args.overwrite:
                    print("Overwrite models.")
                else:
                    raise ValueError("{} already exists. If you continue to run, set --overwrite to be True.".format(model_path))
            
            self.start_epoch = 0

            self.best_loss = float('infinity')
        
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(package['state_dict'])
        else:
            self.model.load_state_dict(package['state_dict'])
        
        self.optimizer.load_state_dict(package['optim_dict'])
    
    def run(self):
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            train_loss, valid_loss = self.run_one_epoch(epoch)
            end = time.time()
            
            print("[Epoch {}/{}] loss (train): {:.5f}, loss (valid): {:.5f}, {:.3f} [sec]".format(epoch + 1, self.epochs, train_loss, valid_loss, end - start), flush=True)
            
            self.train_loss[epoch] = train_loss
            self.valid_loss[epoch] = valid_loss
            
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                model_path = os.path.join(self.model_dir, "finetuned_best.pth")
                self.save_model(epoch, model_path)
            
            model_path = os.path.join(self.model_dir, "finetuned_last.pth")
            self.save_model(epoch, model_path)
            
            save_path = os.path.join(self.loss_dir, "loss.png")
            draw_loss_curve(train_loss=self.train_loss[:epoch+1], valid_loss=self.valid_loss[:epoch+1], save_path=save_path)

    def run_one_epoch_train(self, epoch):
        """
        Training
        """
        self.model.train()
        
        n_sources = self.n_sources
        train_loss = 0
        n_train_batch = len(self.train_loader)
        
        for idx, (mixture, sources) in enumerate(self.train_loader):
            if self.use_cuda:
                mixture = mixture.cuda()
                sources = sources.cuda()
            
            batch_size = sources.size(0)
            accumlated_loss = 0

            for stage_idx in range(n_sources - 1):
                estimated_sources = self.model(mixture)
                loss, indices = self.pit_criterion(estimated_sources, sources)
                accumlated_loss = accumlated_loss + loss

                sources_rest = []

                for batch_idx in range(batch_size):
                    possible_indices = list(range(n_sources - stage_idx))
                    idx_one = indices[batch_idx].item()
                    del possible_indices[idx_one]
                    
                    _sources_rest = torch.index_select(sources[batch_idx], dim=0, index=possible_indices)
                    sources_rest.append(_sources_rest)
                
                _, mixture = torch.split([1, n_sources - stage_idx - 1])
                sources = torch.cat(sources_rest, dim=0)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.max_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            
            self.optimizer.step()
            
            train_loss += loss.item()
            
            if (idx + 1)%100 == 0:
                print("[Epoch {}/{}] iter {}/{} loss: {:.5f}".format(epoch + 1, self.epochs, idx + 1, n_train_batch, loss.item()), flush=True)
        
        train_loss /= n_train_batch
        
        return train_loss

    def run_one_epoch_eval(self, epoch):
        """
        Validation
        """
        self.model.eval()

        n_sources = self.n_sources
        valid_loss = 0
        n_valid = len(self.valid_loader.dataset)
        
        with torch.no_grad():
            for idx, (mixture, sources, segment_IDs) in enumerate(self.valid_loader):
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                
                output_one_and_rest = self.model(mixture)
                output_one = output_one_and_rest[:, 0:1]
                output_rest = output_one_and_rest[:, 1:]
                output = output_one

                for source_idx in range(1, n_sources - 1):
                    output_one_and_rest = self.model(output_rest)
                    output_one = output_one_and_rest[:, 0:1]
                    output_rest = output_one_and_rest[:, 1:]
                    output = torch.cat([output, output_one], dim=1)
                
                output = torch.cat([output, output_rest], dim=1)
                loss, _ = self.pit_criterion(output, sources, batch_mean=False)
                loss = loss.sum(dim=0)
                valid_loss += loss.item()
                
                if idx < 5:
                    mixture = mixture[0].squeeze(dim=0).detach().cpu().numpy()
                    estimated_sources = output[0].detach().cpu().numpy()
                    
                    save_dir = os.path.join(self.sample_dir, segment_IDs[0])
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "mixture.wav")
                    norm = np.abs(mixture).max()
                    mixture = mixture / norm
                    write_wav(save_path, signal=mixture, sr=self.sr)
                    
                    for source_idx, estimated_source in enumerate(estimated_sources):
                        save_path = os.path.join(save_dir, "epoch{}-{}.wav".format(epoch + 1, source_idx + 1))
                        norm = np.abs(estimated_source).max()
                        estimated_source = estimated_source / norm
                        write_wav(save_path, signal=estimated_source, sr=self.sr)
        
        valid_loss /= n_valid

        return valid_loss

class AdhocFinetuneTrainer(FinetuneTrainer):
    def __init__(self, model, loader, pit_criterion, optimizer, args):
        super().__init__(model, loader, pit_criterion, optimizer, args)