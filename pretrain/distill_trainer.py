import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader import BERT4ETHDataloader
from modeling import BERT4ETH, DISTILBERT4ETH
import pickle as pkl
from vocab import FreqVocab
from config import args
# from trainer import BERT4ETHTrainer



class DistillationTrainer:
    def __init__(
        self,
        teacher_model,
        student_model,
        train_loader,
        val_loader,
        temperature=2.0,
        alpha=0.5,
        learning_rate=0.001
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher_model = teacher_model.to(self.device)
        self.student_model = student_model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.temperature = temperature
        self.alpha = alpha
        
        self.optimizer = torch.optim.Adam(self.student_model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def distillation_loss(self, student_logits, teacher_logits):
        """
        Compute the knowledge distillation loss using KL divergence
        """
        T = self.temperature
        soft_targets = F.softmax(teacher_logits / T, dim=1)
        student_log_softmax = F.log_softmax(student_logits / T, dim=1)
        
        return nn.KLDivLoss(reduction='batchmean')(student_log_softmax, soft_targets) * (T * T)
    
    def train_epoch(self):
        self.student_model.train()
        self.teacher_model.eval()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader)):
            data, target = data.to(self.device), target.to(self.device)
            
            # Get teacher predictions
            with torch.no_grad():
                teacher_logits = self.teacher_model(data)
            
            # Train student
            self.optimizer.zero_grad()
            student_logits = self.student_model(data)
            
            # Calculate losses
            student_loss = self.criterion(student_logits, target)
            distillation_loss = self.distillation_loss(student_logits, teacher_logits)
            
            # Combine losses
            loss = (self.alpha * student_loss) + ((1 - self.alpha) * distillation_loss)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self, loader):
        self.student_model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.student_model(data)
                pred = output.argmax(dim=1)
                total_correct += pred.eq(target).sum().item()
                total_samples += target.size(0)
        
        return total_correct / total_samples
    
    def train(self, epochs):
        best_acc = 0
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_acc = self.evaluate(self.val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Accuracy: {val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.student_model.state_dict(), 'best_student_model.pth')
            
            print("-" * 50)

# Example usage:
def main():
    teacher_model = BERT4ETH()
    teacher_model.load_state_dict(torch.load('teacher_model.pth'))
    student_model = DISTILBERT4ETH()
    
    vocab = FreqVocab()
    print("===========Load Sequence===========")
    with open(args.data_dir + "eoa2seq_" + args.bizdate + ".pkl", "rb") as f:
        eoa2seq = pkl.load(f)

    print("number of target user account:", len(eoa2seq))
    vocab.update(eoa2seq)
    # generate mapping
    vocab.generate_vocab()

    # save vocab
    print("token_size:{}".format(len(vocab.vocab_words)))
    vocab_file_name = args.data_dir + args.vocab_filename + "." + args.bizdate
    print('vocab pickle file: ' + vocab_file_name)
    with open(vocab_file_name, 'wb') as output_file:
        pkl.dump(vocab, output_file, protocol=2)

    train_loader = BERT4ETHDataloader(args, vocab, eoa2seq)
    # val_loader = BERT4ETHDataloader(val_dataset, batch_size=32)
    
    # Initialize trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_loader=train_loader,
        # val_loader=val_loader,
        temperature=2.0,
        alpha=0.5,
        learning_rate=0.001
    )
    
    # Train the student model
    trainer.train(epochs=10)

if __name__ == "__main__":
    main()