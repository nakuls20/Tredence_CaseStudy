# Tredence_CaseStudy
# Self-Pruning Neural Networks via Gated Sparsity (CIFAR-10)

This repository implements a differentiable model pruning pipeline on the CIFAR-10 dataset. By utilizing learnable sigmoid gates and research-backed optimization strategies, the model is capable of achieving >80% sparsity with minimal impact on classification accuracy.

## 🚀 Quick Start

### 1. Installation
Ensure you have Python 3.8+ installed. Clone this repository and install the dependencies:

```bash
# Clone the repository
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

# Install required libraries
pip install -r requirements.txt

Run the main script to start the training process.
python tpprunemodel.py
