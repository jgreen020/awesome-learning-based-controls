# Learning Based Controls
Learning-Based Controls is an emerging field at the intersection of Control Theory for Dynamic Systems and Machine Learning, particularly Deep Learning and Reinforcement Learning. This repository serves to curate resources relevant to the field, including introductory information on ML and Controls, topics in LBC, applications of LBC, tools for implementing LBCs, and more.

This repository was originally developed by Jacob Colwell as a student project for courses in Learning-Based Controls (MECHENG 6193, MECHENG 7194) taught at The Ohio State University in the Spring and Autumn semesters of 2025 by Dr. Qadeer Ahmed and Dr. Sidra Ghayour Bhatti.
``` mermaid
mindmap
)__Learning Based Controls__(
	((Control Theory))
		Modeling
			State Space
			Transfer Functions
			Frequency Domain
			Block Diagram
			Feedback
		Design and Analysis
			System Response
				Steady State
				Transient
				Frequency
			Stability
				Lyapunov
				Asymtotic
				Exponential
				Marginal
				BIBO
				Contraction Theory
			Observability
			Controllability
			Pole Placement
		Controllers
			Continuous
				PID
			Discreet
			Adaptive
			Fuzzy-Logic
			Robust
			Optimal
				Conditions
				Linear Quadratic Regulator
				Model Predictive Control
			Stochastic
			Multi-Agent Cooperative Control
		Observers
			Separability Principle
	((Machine Learning))
		Deep Learning
			Architectures
				Physics-Informed Neural Networks
				Feedforward Neural Network
				Recurrent Nerual Networks
				Long Short-Term Memory
				Transformers
				Autoencoders
				Generative Adversarial Networks
				Graph Neural Networks
				Mamba
				Liquid Neural Networks
				Spiking Nerual Networks
				Back Propagation
			Hyperparameters
				Activation Functions
					Sigmoid
					Tanh
					ReLU
					ELU
					Swish
					Leaky ReLU
					Softmax
				Weight Initialization
				Epochs
				Width
				Depth
				Loss Functions
				Optimizers
					Vanilla
					Batch Methods
					Batch Size
					Stochastic Methods
					Momentum
				Learning Rate Schedule
		Learning Paradigms
			Supervised
				Regression
					Linear
					Polynomial
					Ridge
					LASSO
					Gaussian Process
				Classification
					Logistic Regression
					Linear Discriminant Analysis
					Naïve Bayes
				Both
					Decision Trees
					Random Forest
					K-Nearest Neighbors
					Support Vector Machines
			Unsupervised
				Clustering
					K-Means
					Polynomial
					Fuzzy C-Means
				Dimensionality Reduction
					Principal Component Analysis
					Kernel Principal Analysis
				Anomaly Detection
			Semi-Supervised
		Trends
			Physical AI
			General AI
			Agentic AI
	((Optimization))
		Cost Functions
			Mean Absolute Error
			Mean Squared Error
			Log-Cosh Loss
			Cross-Entropy
			KL-Divergence
			Regularization Loss
		Global vs Local Optima
		Unconstrained Optimization
			Gradient Descent
				Mirrored Descent
			Newton's Method
			Stochastic Gradient Descent
			Stochatic Gradient Descent with Momentum
			Adaptive Gradient
			Adaptive Delta
			Adaptive Moment Estimation
		Constrained Optimization
			Equality Constraints
			Inequality Constraints
		Exploration vs. Exploitation
		Trust Region
	((Machine Learning For Control))
		System Identification
			Physics-Informed Neural Networks
			Koopman Operator
				Dynamic Mode Decomposition
		GenAI
			State Estimation
			Trajectory Generation
		Modeling Uncertainty
		ML Controllers
			NN Controller
			GP-Enhanced MPC
		Controller Tuning using RL
		Reinforcement Learning
			Essential Concepts
				Markov Decision Process
				State
				Action
				Reward
				Policy Functions
					Stochastic vs. Deterministic
				Value Functions
				State-Action Function
				Bellman Optimality
				Advantage
				Replay Buffer
				Model Based vs. Model Free
				Policy Based vs. Value Based
				On Policy vs. Off Policy
				Gradient Based vs. Gradient-Free
			Model Based
				Dynamic Programming
					Policy Evaluation
					Policy Iteration
					Value Iteration
				Dyna
					Dyna-Q
				Monte Carlo Search
					Monte Carlo Tree Search
				Partially Observable MDP
			Model Free
				Monte Carlo Methods
					First Visit
					MC Control
				Temporal Difference
					TD Prediction
					TD Control
				SARSA
				Q-Learning
					Deep Q Network
					Double Deep Q Network
					Implicit Q Learning
					Transformer Q Learning
				Actor-Critic
					Advantage Actor Critic
					Deep Deterministic Policy Gradient
					Twin Delayed Deep Deterministic Policy Gradient
					Soft Actor-Critic
				Descision Transformers
				Policy Optimization
					Policy Gradient
					REINFORCE
					Proximal Policy Optimization
					Trust Region Policy Optimization
					Group Relative Policy Optimization
					Teacher Action Distillation with Policy Optimization
			Adversarial Reinforcement Learning
				Imitation Learning
			Multi-Agent Reinforcement Learning
				Asynchronous Advantage Actor-Critic
	((Control For Machine Learning))
		Lipschitz Constraints
		Reachability Analysis
		Quadratic Constraints
		Contraction Theory
		Predictive Safety Filter
		Safe RL with MPC
	((Regulation))
		Guidance for AI ISO-8800
		Functional Safety ISO 26262
		Safety of the Intended Functionality ISO 21448
		Additional Guidance for Autonomy UL 4600
```
## Table of Contents
- [Learning Based Controls](#learning-based-controls)
	- [Table of Contents](#table-of-contents)
	- [Background](#background)
		- [Control Theory](#control-theory)
		- [Machine Learning (Non RL)](#machine-learning-non-rl)
		- [Optimization](#optimization)
	- [Machine Learning For Control](#machine-learning-for-control)
		- [System Identification](#system-identification)
		- [ML Observers](#ml-observers)
		- [ML for Trajectory Generation](#ml-for-trajectory-generation)
		- [ML Controllers](#ml-controllers)
		- [Controller Tuning](#controller-tuning)
		- [Reinforcement Learning](#reinforcement-learning)
	- [Control For Machine Learning](#control-for-machine-learning)
	- [Regulation](#regulation)
	- [Tools](#tools)
		- [General Tools for Machine Learning or Control](#general-tools-for-machine-learning-or-control)
		- [Specific Tools for Learning-Based Controls](#specific-tools-for-learning-based-controls)
	- [Tutorials](#tutorials)
	- [Applications](#applications)
	- [Further Reading](#further-reading)
		- [Awesome Lists](#awesome-lists)
		- [Lectures / Courses](#lectures--courses)
		- [Textbooks](#textbooks)
## Background
### [Control Theory](https://archive.org/details/systemdynamics0000palm_h7i2/mode/2up)
Control Theory is one of the foundational fields from which Learning-Based Control Arises. The following is a non-comprehensive list of concepts from Control Theory that will be important to understand before exploring concepts in Learning-Based Control. Most of these concepts can be explored in the linked textbook, unless otherwise linked. See [Awesome Control Theory](https://github.com/A-make/awesome-control-theory) for a more comprehensive list of topics, resources, and tools.
- Essential Concepts
	- Modeling
		- State Space Representations
		- Transfer Functions
		- Frequency Domain
		- Block Diagrams
	- System Response
	    - Steady State
	    - Transient
	    - Frequency
	- Stability
	    - Lyapunov
	    - Asymtotic
	    - Exponential
	    - Marginal
	    - BIBO
	    - Contraction Theory
	- Observability
	- Controllability
	- Pole Placement
- Control Strategies / Subfields
	- PID Control
	- [Adaptive Control](https://www.researchgate.net/publication/299747127_Robust_Adaptive_Control)
	- Fuzzy-Logic Control
	- [Robust Control](https://archive.org/details/isbn_9788131718872/page/n7/mode/2up)
	- [Optimal Control](https://archive.org/details/optimalcontrolth0000kirk/page/n9/mode/2up)
    - [Model Predictive Control](https://sites.engineering.ucsb.edu/~jbraw/mpc/)
	- Stochastic Control
	- Multi-Agent Cooperative Control
- Observers
	- Separability Principle
### [Machine Learning (Non RL)](https://www.statlearning.com)
It is also important to understand the fundamentals of machine learning and deep learning. Exploring non-deep methods before deep ones may help motivate the deep approaches. As before, these concepts can be explored in the linked textbook, unless otherwise linked. See [Awesome Deep Learning](https://github.com/ChristosChristofidis/awesome-deep-learning#readme) for a more comprehensive list of topics, resources, and tools.
- Traditional Machine Learning
	- Supervised Learning
		- Regression
			- Linear
			- Polynomial
			- Ridge
			- LASSO
			- Gaussian Process
		- Classification
			- Logistic Regression
			- Linear Discriminant Analysis
			- Naïve Bayes
		- Both
			- Decision Trees
			- Random Forest
			- K-Nearest Neighbors
			- Support Vector Machines
	- Unsupervised Learning
		- Clustering
			- K-Means
			- Polynomial
			- Fuzzy C-Means
		- Dimensionality Reduction
			- Principal Component Analysis
			- Kernel Principal Analysis
		- Anomaly Detection
	- Semi-Supervised Learning
	- Loss Functions
    	- Mean Absolute Error
    	- Mean Squared Error
    	- Log-Cosh Loss
    	- Cross-Entropy
    	- KL-Divergence
    	- Regularization Loss
- Deep Learning
	- Architectures
		- Feedforward Neural Network
		- Physics-Informed Neural Networks
		- Recurrent Neural Networks
		- Long Short-Term Memory
		- Transformers
		- Autoencoders
		- Generative Adversarial Networks
		- Graph Neural Networks
		- Mamba
		- Liquid Neural Networks
		- Spiking Nerual Networks
    - Hyperparameters
    	- Activation Functions
    		- Sigmoid
    		- Tanh
    		- ReLU
    		- ELU
    		- Swish
    		- Leaky ReLU
    		- Softmax
    	- Weight Initialization
    	- Epochs
    	- Width
    	- Depth
    	- Loss Functions
    	- Optimizers (See [Optimization](#Optimization))
    	- Learning Rate Schedule
- Trends
	- Physical AI
	- General AI
	- Agentic AI
### [Optimization](https://web.stanford.edu/class/ee364a/)
Mathematical Optimization is fundamental to topics in Machine Learning and Optimal Control. Advanced topics from optimization are employed in Learning-Based Controls Techniques, so it is useful to expand on them here. See [Awesome Optimization](https://github.com/ebrahimpichka/awesome-optimization) for a more comprehensive list of topics, resources, and tools.
- Convexity
- Global vs Local Optima
- Unconstrained Optimization
	- Gradient Descent
	- Newton's Method
	- [Stochastic Gradient Descent (SGD)](https://probml.github.io/pml-book/book1.html)
	- [Stochatic Gradient Descent with Momentum](https://www.nature.com/articles/323533a0)
	- [Adaptive Gradient (AdaGrad)](https://jmlr.org/papers/volume12/duchi11a/duchi11a.pdfat)
	- [Adaptive Moment Estimation (ADAM)](https://arxiv.org/abs/1412.6980)
	- [Mirrored Descent](https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15850-f20/www/notes/lec19.pdf)
- Constrained Optimization
	- Equality Constraints
	- Inequality Constraints
- Exploration vs. Exploitation
- Trust Region Optimization
## Machine Learning For Control
Learning-Based Control is the intersection of Machine Learning and Control Theory, so topics in Learning-Based Control lie on a spectrum between the two fields. We generally split this spectrum into two groups. Topics that involve implementing Machine Learning into the control loop (_e.g._ estimating an element in the control loop) or using Machine Learning Models as controllers (_e.g._ Neural Network Controllers, Reinforcement Learning) are categorized as __Machine Learning for Control__.
### System Identification
System Identification is the process of modeling Dynamic Systems based on data. In the most simple cases (1st- or 2nd-order aproximately-linear systems) it is possible to accurately model them using simple white-box models. For more complicated systems, more advanced statistical models can be used to learn the behavior of the dynamic system, including Deep Learning methods. Using probabilistic machine learning models, we can model uncertainties and/or disturbances in the plant, allowing for more robust controller design with stability and safety guarantees.
- Physics-Informed Neural Networks
- Koopman Operator
	- Dynamic Mode Decomposition
### ML Observers
In systems that are not fully observable, ML methods can be applied in order to reconstruct states required for feedback control. These ML-based-observers can be highly-flexible and reconstruct far more complex state spaces than previously possible with white- or grey-box models.
### ML for Trajectory Generation
Autonomous systems may already have well-performing controllers for motion control (_e.g._ 6-DoF Robots) but lack the ability to plan reference trajectories to perform desired tasks. Reinforcement Learning is a possible solution, but it requires bespoke reward design. Methods based on LLMs have emerged to allow for natural language commands to be interpreted by an autonomous system and converted into reference trajectories to allow them to perform generalized tasks.
### ML Controllers
Controllers can be replaced with ML models, or ML models can be implemented into existing controllers, in order to produce more-complex non-linear controllers, allowing for learning of more precise control strategies from data. 
- NN Controller
- GP-Enhanced MPC
### Controller Tuning
Tuning of controller gains even in simple controllers can be challenging, essentially equivalent to an optimization problem over the space of possible parameter sets. Using different machine learning methods, optimal sets of gains can be found.
### Reinforcement Learning
Reinforcement Learning and Optimal Control are heavily overlapping subfields. They share a significant amount of their problem space, meaning that RL can be applied to most dynamic systems under the right conditions. RL has shown great success at identifying control policies for highly complex systems and thus is an essential part of Machine Learning for Control.
- Essential Concepts
	- Markov Decision Process
	- States, Actions, Rewards, Returns
	- Policy Functions
	- Stochastic vs. Deterministic
	- Value Functions
	- State-Action Functions
	- Bellman Optimality
	- Advantage
	- Replay Buffer
	- Model-Based vs. Model-Free
	- Policy-Based vs. Value-Based
	- On-Policy vs. Off-Policy
	- Gradient-Based vs. Gradient-Free
- Methods
	- Model-Based Methods
	    - Dynamic Programming
	    	- Policy Evaluation
	    	- Policy Iteration
	    	- Value Iteration
	    - Dyna
	    	- Dyna-Q
	    - Monte Carlo Search
	    	- Monte Carlo Tree Search
	    - Partially Observable MDP
	- Model-Free Methods
	    - Monte Carlo Methods
	    	- First Visit
	    	- MC Control
	    - Temporal Difference
	    	- TD Prediction
	    	- TD Control
	    - SARSA
	    - Q-Learning
	    	- Deep Q Network
	    	- Double Deep Q Network
	    	- Implicit Q Learning
	    	- Transformer Q Learning
	    - Actor-Critic
	    	- Advantage Actor Critic
	    	- Deep Deterministic Policy Gradient
	    	- Twin Delayed Deep Deterministic Policy Gradient
	    	- Soft Actor-Critic
	    - Descision Transformers
	    - Policy Optimization
	    	- Policy Gradient
	    	- REINFORCE
	    	- Proximal Policy Optimization
	    	- Trust Region Policy Optimization
	    	- Group Relative Policy Optimization
	    	- Teacher Action Distillation with Policy Optimization
- Adversarial Reinforcement Learning
- Imitation Learning
- Multi-Agent Reinforcement Learning
	- Asynchronous Advantage Actor-Critic
## Control For Machine Learning
Learning-Based Control is the intersection of Machine Learning and Control Theory, so topics in Learning-Based Control lie on a spectrum between the two fields. We generally split this spectrum into two groups. Topics that involve applying Control Theory to Machine Learning (_e.g._ developing notions of stability for model training) are categorized as __Control for Machine Learning__
- Lipschitz Constraints
- Reachability Analysis
- Quadratic Constraints
- Contraction Theory
- Predictive Safety Filter
- Safe RL with MPC
## Regulation
While regulation on the use of AI and ML in general commercial products is not currently implemented, individual regulatory bodies are beginning to pass guidance for the development and implementation of AI systems.
- Autonomous Vehicles
	- [ISO-8800: Road vehicles — Safety and artificial intelligence](https://www.iso.org/standard/83303.html)
	- [ISO 26262: Road vehicles - Functional Safety](https://www.iso.org/standard/68383.html)
	- [ISO 21448: Road vehicles - Safety of the intended functionality](https://www.iso.org/standard/77490.html)
    - [UL 4600: Standard for Evaluation of Autonomous Products](https://www.shopulstandards.com/ProductDetail.aspx?productId=UL4600_3_S_20230317)
- Robotics and Automation
    - [1872-2015: IEEE Standard Ontologies for Robotics and Automation](https://standards.ieee.org/ieee/1872/5354/)
    - [1872.2-2021: Standard for Autonomous Robotics (AuR) Ontology](https://standards.ieee.org/ieee/1872.2/7094/)
    - [7007-2021: Ontological Standard for Ethically Driven Robotics and Automation Systems](https://standards.ieee.org/ieee/7007/7070/)
- General Machine Learning / AI Systems
    - [ISO/IEC 22989: Information technology — Artificial intelligence — Artificial intelligence concepts and terminology](https://www.iso.org/standard/74296.html)
    - [ISO/IEC 23053: Framework for Artificial Intelligence (AI) Systems Using Machine Learning (ML)](https://www.iso.org/standard/74438.html)
    - [ISO/IEC 23894: Information technology — Artificial intelligence — Guidance on risk management](https://www.iso.org/standard/77304.html)
    - [ISO/IEC 42001: Information technology — Artificial intelligence — Management system](https://www.iso.org/standard/42001)
    - [OECD AI Principles](https://oecd.ai/en/ai-principles)
    - [EU Artificial intelligence act](https://artificialintelligenceact.eu/ai-act-explorer/)
    - [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
## Tools
Implementation, Testing, and Development of LBC systems can be performed using a variety of environments, packages, and hardware.
### General Tools for Machine Learning or Control
Many platforms are available for machine learning and control, and are well summarized by other awesome lists.
- Software
    - [Awesome Machine Learning](https://github.com/josephmisiti/awesome-machine-learning#readme)
    - [Awesome Python Data Science](https://github.com/krzjoa/awesome-python-data-science#readme)
    - [Awesome Control Theory](https://github.com/A-make/awesome-control-theory?tab=readme-ov-file#tools)
- Hardware
	- [Awesome Robotics](https://github.com/Kiloreux/awesome-robotics#readme)
### Specific Tools for Learning-Based Controls
The following are tools that have bespoke tools for combining learning and controls

## Tutorials
Great tutorials on various ML, Controls, and LBS topics have been developed. So for more hands-on involvment in these methods, see the following tutorials. Several notebooks have also been developed as part of this course to demonstrate various LBC methods so they will be included here.
- [Awesome Machine Learning Tutorials](https://github.com/ujjwalkarn/Machine-Learning-Tutorials#readme)
## Applications
LBC systems have been applied in various fields. The following is a list of repositories that demonstrate use of LBC methods. Primarily, this list contains projects that were developed as part of the aforementioned courses.
## Further Reading
Other documents/videos that might be of interest to those interested in Learning-Based Controls.
### Awesome Lists
- [Awesome Deep Learning Papers](https://github.com/terryum/awesome-deep-learning-papers)
- [Awesome Deep Learning](https://github.com/ChristosChristofidis/awesome-deep-learning#readme)
- [Awesome Complexity](https://github.com/sellisd/awesome-complexity#readme)
- [Awesome Control Theory](https://github.com/A-make/awesome-control-theory)
- [Awesome Decision Transformers](https://github.com/opendilab/awesome-decision-transformer)
- [Awesome Model-Based RL](https://github.com/opendilab/awesome-model-based-RL)
- [Awesome Offline RL](https://github.com/hanjuku-kaso/awesome-offline-rl)
- [Awesome RLHF](https://github.com/opendilab/awesome-RLHF)
- [Awesome End-to-End Autonomous Driving](https://github.com/opendilab/awesome-end-to-end-autonomous-driving)
- [Awesome Legged Locomotion Learning](https://github.com/gaiyi7788/awesome-legged-locomotion-learning)
- [Awesome Driving Behavior Prediciton](https://github.com/opendilab/awesome-driving-behavior-prediction)
- [Reinforcement Learning Resources](https://github.com/datascienceid/reinforcement-learning-resources)
### Lectures / Courses
- [Artemis Lab LBC Lectures](https://www.youtube.com/playlist?list=PLqCrLscdNVX9lmstqRoI6nd22Stwflmtx)
- MATLAB Tech Talks
	- [Data-Driven Control](https://www.mathworks.com/videos/series/learning-based-control.html)
	- [Reinforcement Learning](https://www.mathworks.com/videos/series/reinforcement-learning.html)
	- [Control Systems in Practice](https://www.mathworks.com/videos/series/control-systems-in-practice.html)
	- [State Space](https://www.mathworks.com/videos/series/state-space.html)
- [AA203: Optimal and Learning-Based Control, Stanford University](https://stanfordasl.github.io/aa203/sp2425)
- [Data Driven Control with Machine Learning by Steve Brunton](https://www.youtube.com/playlist?list=PLMrJAkhIeNNQkv98vuPjO2X2qJO_UPeWR)
- [Reinforcement Learning by Steve Brunton](https://www.youtube.com/playlist?list=PLMrJAkhIeNNQe1JXNvaFvURxGY4gE9k74)
### Textbooks
- [Learning-Based Control: A Tutorial and Some Recent Results](http://dx.doi.org/10.1561/2600000023)