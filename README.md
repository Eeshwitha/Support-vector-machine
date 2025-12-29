SVM Task – Parts 1 to 5

This repository contains the Python code and explanations for an SVM (Support Vector Machine) task with five parts.

Part 1: Kernel Selection

Dataset: Ring-shaped data with 2 features.
Kernel used: RBF (Radial Basis Function) kernel.
Reason: RBF allows SVM to separate non-linear data without explicitly creating new features.
Observation: Scatter plot shows positive class forming a ring around the negative class.

Part 2: Hyperparameter Effects

C values tested: 1.0 and 10.0
Effect of increasing C:
Margin width decreases
Model becomes more sensitive to individual points
Number of support vectors may increase or decrease depending on dataset
Observed support vectors:
C=1 → [17,17]
C=10 → [9,4]

Part 3: Support Vector Properties

Only points near the decision boundary become support vectors.
Points far from the boundary, even duplicates, do not affect the SVM.
Observed support vectors: [16,13]
Optional step: Removing far-away duplicates does not change support vectors.

Part 4: Feature Scaling

Without scaling: Large-scale features dominate, SVM misbehaves → [30,31] support vectors.
With scaling: Features contribute equally, SVM behaves correctly → [16,13] support vectors.
Conclusion: Scaling is critical because SVM uses distances to separate classes.

Part 5: Decision Function

Weight vector: [0.5, -1.2], Bias: -0.3, Input: [4,3]
Decision function value: -1.9
Predicted class: Negative
Explanation: f(x) = w·x + b → -1.9 < 0 → Negative class.
Summary

All parts demonstrate key SVM concepts: kernel trick, hyperparameter effects, support vectors, feature scaling, and decision function.
