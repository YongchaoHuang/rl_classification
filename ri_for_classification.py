# -*- coding: utf-8 -*-
"""RL for classification.ipynb
# yongchao.huang@abdn.ac.uk

# Stage 1: a working but not good example using REINFORCE.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import time
import random

# --- 1. Setup, Seeds, and Hyperparameters ---
# Set a seed for full reproducibility
SEED = 111
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# For full reproducibility on CUDA, if available
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LEARNING_RATE = 0.01
EPOCHS = 600
GAMMA = 0.99 # Discount factor, though less critical in this single-step setting

# --- 2. Load and Prepare the Iris Dataset ---
# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names
n_classes = len(class_names)

# Binarize the output for ROC curve calculation
y_bin = label_binarize(y, classes=[0, 1, 2])

# Split data into training and testing sets
X_train, X_test, y_train, y_test, y_train_bin, y_test_bin = train_test_split(
    X, y, y_bin, test_size=0.2, random_state=SEED, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).long().to(device)
X_test_tensor = torch.from_numpy(X_test).float().to(device)
y_test_tensor = torch.from_numpy(y_test).long().to(device)

# --- 3. Define the Policy Network (The RL Classifier) ---
class Policy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Policy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

# --- 4. The REINFORCE Agent ---
class ReinforceAgent:
    def __init__(self, input_dim, output_dim, lr=LEARNING_RATE):
        self.policy = Policy(input_dim, output_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.item()

    def update_policy(self):
        returns = []
        discounted_reward = 0
        for r in self.rewards[::-1]:
            discounted_reward = r + GAMMA * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, device=device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append((-log_prob * R).unsqueeze(0))

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []

# --- 5. Training Loop ---
input_dim = X_train.shape[1]
output_dim = n_classes
agent = ReinforceAgent(input_dim, output_dim)

all_rewards = []
print("Starting training...")
start_time = time.time()

for epoch in range(EPOCHS):
    total_reward = 0
    for i in range(len(X_train_tensor)):
        state = X_train_tensor[i]
        true_label = y_train_tensor[i].item()
        action = agent.select_action(state)
        reward = 1 if action == true_label else 0
        agent.rewards.append(reward)
        total_reward += reward

    agent.update_policy()
    avg_reward = total_reward / len(X_train_tensor)
    all_rewards.append(avg_reward)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Reward: {avg_reward:.4f}")

print("Training finished.")
end_time = time.time()
total_time = end_time - start_time

# --- 6. Evaluation and Visualization ---
y_pred = []
y_pred_probs = []
with torch.no_grad():
    for i in range(len(X_test_tensor)):
        state = X_test_tensor[i]
        probs = agent.policy(state)
        action = torch.argmax(probs).item()
        y_pred.append(action)
        y_pred_probs.append(probs.cpu().numpy())

y_pred_probs = np.array(y_pred_probs)

# Print Classification Report
print("\n" + "="*50)
print("Classification Report")
print("="*50)
report = classification_report(y_test, y_pred, target_names=class_names)
print(report)
print(f"Total Execution Time: {total_time:.2f} seconds")
print("="*50 + "\n")


# Create a 1x4 grid for plots
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
fig.suptitle('RL Classifier Performance Evaluation on Iris Dataset', fontsize=20)

# Plot 1: Learning Curve
axes[0].plot(all_rewards)
axes[0].set_title("Learning Curve (Average Reward)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Average Reward per Epoch")
axes[0].grid(True)
axes[0].set_ylim(0, 1.05)

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title("Confusion Matrix")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

# Plot 3: ROC Curves (One-vs-Rest)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

colors = ['aqua', 'darkorange', 'cornflowerblue']
for i, color in zip(range(n_classes), colors):
    axes[2].plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')

axes[2].plot([0, 1], [0, 1], 'k--', lw=2)
axes[2].set_xlim([0.0, 1.0])
axes[2].set_ylim([0.0, 1.05])
axes[2].set_xlabel('False Positive Rate')
axes[2].set_ylabel('True Positive Rate')
axes[2].set_title('Multi-class ROC Curves')
axes[2].legend(loc="lower right")
axes[2].grid(True)

# Plot 4: Display Classification Report as text
axes[3].axis('off')
axes[3].text(0.05, 0.95, "Classification Report (test set)", fontsize=14, weight='bold', va='top')
axes[3].text(0.05, 0.85, report, family='monospace', fontsize=12, va='top')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

"""# Stage 2: Actor-Critic (A2C)."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import time
import random

# --- 1. Setup, Seeds, and Hyperparameters ---
# Set a seed for full reproducibility
SEED = 111
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# For full reproducibility on CUDA, if available
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
ACTOR_LR = 0.005
CRITIC_LR = 0.01
EPOCHS = 600
GAMMA = 0.99 # Discount factor

# --- 2. Load and Prepare the Iris Dataset ---
# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names
n_classes = len(class_names)

# Binarize the output for ROC curve calculation
y_bin = label_binarize(y, classes=[0, 1, 2])

# Split data into training and testing sets
X_train, X_test, y_train, y_test, y_train_bin, y_test_bin = train_test_split(
    X, y, y_bin, test_size=0.2, random_state=SEED, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).long().to(device)
X_test_tensor = torch.from_numpy(X_test).float().to(device)
y_test_tensor = torch.from_numpy(y_test).long().to(device)

# --- 3. Define the Actor and Critic Networks ---
# The Actor (Policy) network outputs action probabilities
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

# The Critic (Value) network estimates the value of a state
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Outputs a single value for the state
        )

    def forward(self, state):
        return self.net(state)

# --- 4. The Actor-Critic Agent ---
class ActorCriticAgent:
    def __init__(self, input_dim, output_dim):
        self.actor = Actor(input_dim, output_dim).to(device)
        self.critic = Critic(input_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

    def update(self, rewards, log_probs, state_values):
        returns = []
        discounted_reward = 0
        for r in rewards[::-1]:
            discounted_reward = r + GAMMA * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, device=device)

        # Calculate Advantage: A(s,a) = R - V(s)
        advantage = returns - state_values

        # Calculate Actor loss
        actor_loss = (-log_probs * advantage.detach()).mean()

        # Calculate Critic loss (MSE between returns and predicted values)
        critic_loss = nn.MSELoss()(state_values, returns)

        # Update Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

# --- 5. Training Loop ---
input_dim = X_train.shape[1]
output_dim = n_classes
agent = ActorCriticAgent(input_dim, output_dim)

all_rewards = []
print("Starting training with Actor-Critic (A2C)...")
start_time = time.time()

for epoch in range(EPOCHS):
    # Store data for the entire epoch before updating
    log_probs = []
    rewards = []
    state_values = []
    total_reward_epoch = 0

    for i in range(len(X_train_tensor)):
        state = X_train_tensor[i]
        true_label = y_train_tensor[i].item()

        # Get action probabilities from the Actor
        action_probs = agent.actor(state)
        m = Categorical(action_probs)
        action = m.sample()

        # Get state value from the Critic
        state_value = agent.critic(state)

        # Store results
        log_probs.append(m.log_prob(action))
        state_values.append(state_value)

        # Determine the reward
        reward = 1 if action.item() == true_label else 0
        rewards.append(reward)
        total_reward_epoch += reward

    # Update the policy after the epoch
    agent.update(
        rewards,
        torch.stack(log_probs),
        torch.cat(state_values)
    )

    avg_reward = total_reward_epoch / len(X_train_tensor)
    all_rewards.append(avg_reward)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Reward: {avg_reward:.4f}")

print("Training finished.")
end_time = time.time()
total_time = end_time - start_time

# --- 6. Evaluation and Visualization ---
y_pred = []
y_pred_probs = []
with torch.no_grad():
    for i in range(len(X_test_tensor)):
        state = X_test_tensor[i]
        probs = agent.actor(state)
        action = torch.argmax(probs).item()
        y_pred.append(action)
        y_pred_probs.append(probs.cpu().numpy())

y_pred_probs = np.array(y_pred_probs)

# Print Classification Report
print("\n" + "="*50)
print("Classification Report")
print("="*50)
report = classification_report(y_test, y_pred, target_names=class_names)
print(report)
print(f"Total Execution Time: {total_time:.2f} seconds")
print("="*50 + "\n")


# Create a 1x4 grid for plots
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
fig.suptitle('A2C Classifier Performance Evaluation on Iris Dataset', fontsize=20)

# Plot 1: Learning Curve
axes[0].plot(all_rewards)
axes[0].set_title("Learning Curve (Average Reward)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Average Reward per Epoch")
axes[0].grid(True)
axes[0].set_ylim(0, 1.05)

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title("Confusion Matrix")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

# Plot 3: ROC Curves (One-vs-Rest)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

colors = ['aqua', 'darkorange', 'cornflowerblue']
for i, color in zip(range(n_classes), colors):
    axes[2].plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')

axes[2].plot([0, 1], [0, 1], 'k--', lw=2)
axes[2].set_xlim([0.0, 1.0])
axes[2].set_ylim([0.0, 1.05])
axes[2].set_xlabel('False Positive Rate')
axes[2].set_ylabel('True Positive Rate')
axes[2].set_title('Multi-class ROC Curves')
axes[2].legend(loc="lower right")
axes[2].grid(True)

# Plot 4: Display Classification Report as text
axes[3].axis('off')
axes[3].text(0.05, 0.95, "Classification Report (test set)", fontsize=14, weight='bold', va='top')
axes[3].text(0.05, 0.85, report, family='monospace', fontsize=12, va='top')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

"""# Stage 3: A2C: customized reward for classification."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import time
import random

# --- 1. Setup, Seeds, and Hyperparameters ---
# Set a seed for full reproducibility
SEED = 111
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# For full reproducibility on CUDA, if available
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
ACTOR_LR = 0.005
CRITIC_LR = 0.01
EPOCHS = 600 # Increased epochs for fair comparison
GAMMA = 0.99 # Discount factor

# --- 2. Load and Prepare the Iris Dataset ---
# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names
n_classes = len(class_names)

# Binarize the output for ROC curve calculation
y_bin = label_binarize(y, classes=[0, 1, 2])

# Split data into training and testing sets
X_train, X_test, y_train, y_test, y_train_bin, y_test_bin = train_test_split(
    X, y, y_bin, test_size=0.2, random_state=SEED, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).long().to(device)
X_test_tensor = torch.from_numpy(X_test).float().to(device)
y_test_tensor = torch.from_numpy(y_test).long().to(device)

# --- 3. Define the Actor and Critic Networks ---
# The Actor (Policy) network outputs action probabilities
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

# The Critic (Value) network estimates the value of a state
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Outputs a single value for the state
        )

    def forward(self, state):
        return self.net(state)

# --- 4. The Actor-Critic Agent ---
class ActorCriticAgent:
    def __init__(self, input_dim, output_dim):
        self.actor = Actor(input_dim, output_dim).to(device)
        self.critic = Critic(input_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

    def update(self, rewards, log_probs, state_values):
        returns = []
        discounted_reward = 0
        for r in rewards[::-1]:
            discounted_reward = r + GAMMA * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, device=device)

        # Calculate Advantage: A(s,a) = R - V(s)
        advantage = returns - state_values

        # Calculate Actor loss
        actor_loss = (-log_probs * advantage.detach()).mean()

        # Calculate Critic loss (MSE between returns and predicted values)
        critic_loss = nn.MSELoss()(state_values, returns)

        # Update Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

# --- 5. Training Loop ---
input_dim = X_train.shape[1]
output_dim = n_classes
agent = ActorCriticAgent(input_dim, output_dim)

all_rewards = []
print("Starting training with Asymmetric Rewards...")
start_time = time.time()

for epoch in range(EPOCHS):
    # Store data for the entire epoch before updating
    log_probs = []
    rewards = []
    state_values = []
    total_reward_epoch = 0

    for i in range(len(X_train_tensor)):
        state = X_train_tensor[i]
        true_label = y_train_tensor[i].item()

        # Get action probabilities from the Actor
        action_probs = agent.actor(state)
        m = Categorical(action_probs)
        action = m.sample()

        # Get state value from the Critic
        state_value = agent.critic(state)

        # Store results
        log_probs.append(m.log_prob(action))
        state_values.append(state_value)

        # *** STAGE 3 CHANGE: ASYMMETRIC REWARD FUNCTION ***
        reward = 0
        predicted_class = action.item()

        if predicted_class == true_label:
            # Higher reward for correctly identifying the important class
            if true_label == 1: # Class 1 is 'versicolor'
                reward = 5
            else:
                reward = 1
        else:
            # Huge penalty for misclassifying the important class
            if true_label == 1: # Misclassified a true 'versicolor'
                reward = -10
            else:
                reward = 0 # No penalty for other errors

        rewards.append(reward)
        total_reward_epoch += reward

    # Update the policy after the epoch
    agent.update(
        rewards,
        torch.stack(log_probs),
        torch.cat(state_values)
    )

    avg_reward = total_reward_epoch / len(X_train_tensor)
    all_rewards.append(avg_reward)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Reward: {avg_reward:.4f}")

print("Training finished.")
end_time = time.time()
total_time = end_time - start_time

# --- 6. Evaluation and Visualization ---
y_pred = []
y_pred_probs = []
with torch.no_grad():
    for i in range(len(X_test_tensor)):
        state = X_test_tensor[i]
        probs = agent.actor(state)
        action = torch.argmax(probs).item()
        y_pred.append(action)
        y_pred_probs.append(probs.cpu().numpy())

y_pred_probs = np.array(y_pred_probs)

# Print Classification Report
print("\n" + "="*50)
print("Classification Report")
print("="*50)
report = classification_report(y_test, y_pred, target_names=class_names)
print(report)
print(f"Total Execution Time: {total_time:.2f} seconds")
print("="*50 + "\n")


# Create a 1x4 grid for plots
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
fig.suptitle('A2C Classifier with Asymmetric Rewards', fontsize=20)

# Plot 1: Learning Curve
axes[0].plot(all_rewards)
axes[0].set_title("Learning Curve (Average Reward)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Average Reward per Epoch")
axes[0].grid(True)
# Note: Y-axis is not limited to 1.05 as rewards can be negative

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title("Confusion Matrix")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

# Plot 3: ROC Curves (One-vs-Rest)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

colors = ['aqua', 'darkorange', 'cornflowerblue']
for i, color in zip(range(n_classes), colors):
    axes[2].plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')

axes[2].plot([0, 1], [0, 1], 'k--', lw=2)
axes[2].set_xlim([0.0, 1.0])
axes[2].set_ylim([0.0, 1.05])
axes[2].set_xlabel('False Positive Rate')
axes[2].set_ylabel('True Positive Rate')
axes[2].set_title('Multi-class ROC Curves')
axes[2].legend(loc="lower right")
axes[2].grid(True)

# Plot 4: Display Classification Report as text
axes[3].axis('off')
axes[3].text(0.05, 0.95, "Classification Report (test set)", fontsize=14, weight='bold', va='top')
axes[3].text(0.05, 0.85, report, family='monospace', fontsize=12, va='top')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

"""# Comparison: logistic regression."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import time
import random

# --- 1. Setup and Seeds ---
# Use the same seed as the RL experiments for a fair comparison
SEED = 111
random.seed(SEED)
np.random.seed(SEED)

# --- 2. Load and Prepare the Iris Dataset ---
# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names
n_classes = len(class_names)

# Binarize the output for ROC curve calculation
y_bin = label_binarize(y, classes=[0, 1, 2])

# Split data into training and testing sets using the same parameters
X_train, X_test, y_train, y_test, y_train_bin, y_test_bin = train_test_split(
    X, y, y_bin, test_size=0.2, random_state=SEED, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- 3. Train the Logistic Regression Model ---
print("Training Logistic Regression model...")
start_time = time.time()

# Initialize and train the model
# 'ovr' for One-vs-Rest, which is standard for multi-class problems
log_reg = LogisticRegression(multi_class='ovr', random_state=SEED)
log_reg.fit(X_train, y_train)

end_time = time.time()
total_time = end_time - start_time
print("Training finished.")

# --- 4. Evaluation and Visualization ---
# Get predictions
y_pred = log_reg.predict(X_test)
# Get prediction probabilities for ROC curve
y_pred_probs = log_reg.predict_proba(X_test)

# Print Classification Report
print("\n" + "="*50)
print("Classification Report")
print("="*50)
report = classification_report(y_test, y_pred, target_names=class_names)
print(report)
print(f"Total Execution Time: {total_time:.4f} seconds")
print("="*50 + "\n")

# Create a 1x3 grid for plots (no learning curve for this model)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Logistic Regression Performance Evaluation on Iris Dataset', fontsize=20)

# Plot 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

# Plot 2: ROC Curves (One-vs-Rest)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

colors = ['aqua', 'darkorange', 'cornflowerblue']
for i, color in zip(range(n_classes), colors):
    axes[1].plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')

axes[1].plot([0, 1], [0, 1], 'k--', lw=2)
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Multi-class ROC Curves')
axes[1].legend(loc="lower right")
axes[1].grid(True)

# Plot 3: Display Classification Report as text
axes[2].axis('off')
axes[2].text(0.05, 0.95, "Classification Report (test set)", fontsize=14, weight='bold', va='top')
axes[2].text(0.05, 0.85, report, family='monospace', fontsize=12, va='top')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
