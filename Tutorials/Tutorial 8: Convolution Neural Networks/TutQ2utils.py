import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from random import randint
import time
# Clear the output and plot updated metrics
from IPython.display import clear_output, display
import torch
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

def learning_rate_scheduler(optimizer, epoch, initial_lr):
    """
    Adjust learning rate according to schedule
    Learning rate is reduced by half after eery 5 epochs
    """
    lr = initial_lr
    if epoch % 5 == 0:
        lr = lr / 2
    return lr

# Calculate and plot ROC curves
def calculate_roc_curves_test(model, x_test, y_test, num_classes, device, mean, std):
    """
    Calculate ROC curves 
    """
   
    # Prepare test data - move to device after normalization
    x_test_float = x_test.float()
    x_test_normalized = (x_test_float - mean.cpu()) / std.cpu()  # Normalize on CPU
    if len(x_test.shape) == 3:
        x_test_normalized = x_test_normalized.unsqueeze(dim=1)  # Add channel dimension
    x_test_normalized = x_test_normalized.to(device)  # Move to device
    
    # Get model predictions
    with torch.no_grad():
        scores = model(x_test_normalized)
    
    # Convert scores to probabilities (move to CPU for sklearn functions)
    probabilities = torch.softmax(scores, dim=1).cpu().detach().numpy()
    
    # Plot ROC curves for all classes
    plt.figure(figsize=(12, 8))
    
    auc_scores = []
    for class_idx in range(num_classes):
        fpr, tpr, _ = roc_curve(y_test.cpu().numpy(), 
                               probabilities[:, class_idx], 
                               pos_label=class_idx)
        auc_score = auc(fpr, tpr)
        auc_scores.append(auc_score)
        
        plt.plot(fpr, tpr, label=f'Digit {class_idx} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for MNIST Digits')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Print AUC scores
    print("\nAUC Scores for each digit:")
    for i, score in enumerate(auc_scores):
        print(f"Digit {i}: {score:.4f}")
    
    avg_auc = sum(auc_scores) / len(auc_scores)
    print(f"\nAverage AUC across all digits: {avg_auc:.4f}")
    
    return auc_scores


def error_rate(scores,labels):
    bs=scores.size(0)
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches=indicator.sum()    
    return 1-num_matches.float()/bs  

def calculate_metrics(model, x_test, y_test, num_classes, device, mean, std):
    """
    Calculate ROC and PR metrics for current epoch
    
    Returns:
    --------
    dict: Dictionary containing all calculated metrics and curves
    """
    model.eval()
    
    # Move y_test to the same device as the model
    y_test = y_test.to(device)

    # Prepare test data
    x_test_float = x_test.float()
    x_test_normalized = (x_test_float - mean.cpu()) / std.cpu()
    if len(x_test.shape) == 3:
        x_test_normalized = x_test_normalized.unsqueeze(dim=1)
    x_test_normalized = x_test_normalized.to(device)
    
    # Get predictions
    with torch.no_grad():
        scores = model(x_test_normalized)
    
    test_error = error_rate(scores,y_test)
    if isinstance(test_error, torch.Tensor):
        test_error = test_error.item()
    probabilities = torch.softmax(scores, dim=1).cpu().detach().numpy()
    y_test_np = y_test.cpu().numpy()
    y_test_onehot = np.eye(num_classes)[y_test_np]
    
    # roc_cruves will store (fpr, tpr) pairs
    # pr_curves will store (precision, recall) pairs
    metrics = {'auc_scores': [], 'ap_scores': [],'roc_curves': [], 'pr_curves': [], 'test_error': test_error }
    
    # Calculate metrics for each class
    for class_idx in range(num_classes):
        # ROC curve metrics
        fpr, tpr, _ = roc_curve(y_test_np, probabilities[:, class_idx], pos_label=class_idx)
        auc_score = auc(fpr, tpr)
        metrics['auc_scores'].append(auc_score)
        metrics['roc_curves'].append((fpr, tpr))
        
        # PR curve metrics
        precision, recall, _ = precision_recall_curve(y_test_onehot[:, class_idx], probabilities[:, class_idx])
        ap = average_precision_score(y_test_onehot[:, class_idx],probabilities[:, class_idx])
        metrics['ap_scores'].append(ap)
        metrics['pr_curves'].append((precision, recall))
    
    # Calculate mean scores
    metrics['mean_auc'] = sum(metrics['auc_scores']) / len(metrics['auc_scores'])
    metrics['mean_ap'] = sum(metrics['ap_scores']) / len(metrics['ap_scores'])
    
    return metrics  

def print_epoch_stats(epoch, elapsed_time, lr, train_loss, train_error, test_error, mean_auc=None, mean_ap=None):
    """
    Print statistics for the current epoch
    """
    print(f'Epoch: {epoch}')
    print(f'Time: {elapsed_time:.2f} min')
    print(f'Learning Rate: {lr}')
    print(f'Training Loss: {train_loss:.4f}')
    print(f'Training Error: {train_error*100:.2f}%')
    print(f'Testing Error: {test_error*100:.2f}%')
    if mean_auc is not None:
        print(f'Mean AUC: {mean_auc:.4f}')
    if mean_ap is not None:
        print(f'Mean AP: {mean_ap:.4f}')
    print('-' * 50)

class DynamicMetricsPlot:
    def __init__(self):
        # Clear any existing plots
        plt.close('all')
        
        # Create figure and subplots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Initialize empty lines
        self.loss_line, = self.ax1.plot([], [], 'b-', label='Training Loss')
        self.train_error_line, = self.ax2.plot([], [], 'g-', label='Training Error')
        self.test_error_line, = self.ax2.plot([], [], 'r-', label='Test Error')
        self.auc_line, = self.ax3.plot([], [], 'b-', label='Mean AUC')
        self.ap_line, = self.ax3.plot([], [], 'r-', label='Mean AP')
        
        # Set up the plots
        self._setup_plots()
        
        # Display the plot
        plt.show()
        
    def _setup_plots(self):
        # Setup for loss plot
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Training Loss vs. Epoch')
        self.ax1.grid(True)
        self.ax1.legend()
        
        # Setup for error plot
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Error Rate')
        self.ax2.set_title('Error Rate vs. Epoch')
        self.ax2.grid(True)
        self.ax2.legend()
        
        # Setup for AUC/AP plot
        self.ax3.set_xlabel('Epoch')
        self.ax3.set_ylabel('Score')
        self.ax3.set_title('AUC and AP vs. Epoch')
        self.ax3.grid(True)
        self.ax3.legend()
        
        plt.tight_layout()

    def update(self, metrics):
        """
        Update the plots with new metrics
        """
        # Clear the current figure
        plt.clf()
        
        # Create new subplots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        epochs = range(1, len(metrics.train_losses) + 1)
        
        # Plot updated data
        # Loss plot
        self.ax1.plot(epochs, metrics.train_losses, 'b-', label='Training Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Training Loss vs. Epoch')
        self.ax1.grid(True)
        self.ax1.legend()
        
        # Error plot
        self.ax2.plot(epochs, metrics.train_errors, 'g-', label='Training Error')
        self.ax2.plot(epochs, metrics.test_errors, 'r-', label='Test Error')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Error Rate')
        self.ax2.set_title('Error Rate vs. Epoch')
        self.ax2.grid(True)
        self.ax2.legend()
        
        # AUC/AP plot
        self.ax3.plot(epochs, metrics.mean_aucs, 'b-', label='Mean AUC')
        self.ax3.plot(epochs, metrics.mean_aps, 'r-', label='Mean AP')
        self.ax3.set_xlabel('Epoch')
        self.ax3.set_ylabel('Score')
        self.ax3.set_title('AUC and AP vs. Epoch')
        self.ax3.grid(True)
        self.ax3.legend()
        
        plt.tight_layout()
        
        # Display the updated plot
        clear_output(wait=True)
        display(self.fig)



def plot_learning_curves(metrics):
    """
    Plot learning curves after training
    """
    plt.figure(figsize=(15, 5))
    
    # Plot training loss
    plt.subplot(1, 3, 1)
    plt.plot(metrics.train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Epoch')
    plt.legend()
    plt.grid(True)
    
    # Plot error rates
    plt.subplot(1, 3, 2)
    plt.plot(metrics.train_errors, label='Training Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate')
    plt.title('Error Rate vs. Epoch')
    plt.legend()
    plt.grid(True)
    
    # Plot AUC and AP
    plt.subplot(1, 3, 3)
    plt.plot(metrics.mean_aucs, label='Mean AUC')
    plt.plot(metrics.mean_aps, label='Mean AP')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('AUC and AP vs. Epoch')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()    

def plot_roc_pr_curves(model, x_test, y_test, num_classes, device, mean, std):
    """
    Plot both ROC and Precision-Recall curves side by side
    """
    plt.close('all')
    # Set model to evaluation mode
    model.eval()
    
    # Prepare test data
    x_test_float = x_test.float()
    x_test_normalized = (x_test_float - mean.cpu()) / std.cpu()
    x_test_normalized = x_test_normalized.unsqueeze(dim=1)
    x_test_normalized = x_test_normalized.to(device)
    
    # Get model predictionsdef plot_roc_pr_curves(model, x_test, y_test, num_classes, device, mean, std):
    """
    Plot both ROC and Precision-Recall curves side by side
    """
    # Clear any existing plots
    plt.clf()
    
    # Set model to evaluation mode
    model.eval()
    
    # Prepare test data
    x_test_float = x_test.float()
    x_test_normalized = (x_test_float - mean.cpu()) / std.cpu()
    x_test_normalized = x_test_normalized.unsqueeze(dim=1)
    x_test_normalized = x_test_normalized.to(device)
    
    # Get model predictions
    with torch.no_grad():
        scores = model(x_test_normalized)
    
    # Convert scores to probabilities
    probabilities = torch.softmax(scores, dim=1).cpu().detach().numpy()
    y_test_np = y_test.cpu().numpy()
    y_test_onehot = np.eye(num_classes)[y_test_np]
    
    # Create new figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    auc_scores = []
    ap_scores = []
    
    # Calculate and plot curves for each class
    for class_idx in range(num_classes):
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test_np, 
                               probabilities[:, class_idx], 
                               pos_label=class_idx)
        auc_score = auc(fpr, tpr)
        auc_scores.append(auc_score)
        ax1.plot(fpr, tpr, label=f'Digit {class_idx} (AUC = {auc_score:.3f})')
        
        # PR curve
        precision, recall, _ = precision_recall_curve(
            y_test_onehot[:, class_idx],
            probabilities[:, class_idx]
        )
        ap = average_precision_score(
            y_test_onehot[:, class_idx],
            probabilities[:, class_idx]
        )
        ap_scores.append(ap)
        ax2.plot(recall, precision, label=f'Digit {class_idx} (AP = {ap:.3f})')
    
    # Customize ROC plot
    ax1.plot([0, 1], [0, 1], 'k--', label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves')
    ax1.grid(True)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Customize PR plot
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves')
    ax2.grid(True)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Print scores
    print("\nScores for each digit:")
    for i in range(num_classes):
        print(f"Digit {i}: AUC = {auc_scores[i]:.4f}, AP = {ap_scores[i]:.4f}")
    
    mean_auc = sum(auc_scores) / len(auc_scores)
    mean_ap = sum(ap_scores) / len(ap_scores)
    print(f"\nMean AUC: {mean_auc:.4f}")
    print(f"Mean AP: {mean_ap:.4f}")
    
    return auc_scores, ap_scores

def show(X):
    if X.dim() == 3 and X.size(0) == 3:
        plt.imshow( np.transpose(  X.numpy() , (1, 2, 0))  )
        plt.show()
    elif X.dim() == 2:
        plt.imshow(   X.numpy() , cmap='gray'  )
        plt.show()
    else:
        print('Wrong Tensor Size')

def show_prob_mnist(p):

    p=p.data.squeeze().numpy()

    ft=15
    label = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight','nine')    
    y_pos = np.arange(len(p))*1.2
    target=2
    width=0.9
    col= 'blue'
    
    plt.rcdefaults()
    fig, ax = plt.subplots()

    # the plot
    ax.barh(y_pos, p, width , align='center', color=col)

    ax.set_xlim([0, 1.3])
   
    ax.set_yticks(y_pos)
    ax.set_yticklabels(label, fontsize=ft)
    ax.invert_yaxis()  
    
    ax.set_xticklabels([])
    ax.set_xticks([])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(4)


    for i in range(len(p)):
        str_nb="{0:.2f}".format(p[i])
        ax.text( p[i] + 0.05 , y_pos[i] ,str_nb ,
                 horizontalalignment='left', verticalalignment='center',
                 transform=ax.transData, color= col,fontsize=ft)
    plt.show()
    




