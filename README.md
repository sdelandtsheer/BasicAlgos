# BasicAlgos
Basic algorithmic exercises in Python

Solutions are provided in full, with insightful comments

Comments, alternative solutions, hacks, etc... welcome

## Contents:

- Missionaries and Cannibals (stochastic and breath-first search): ok
- Fibonacci (multiple techniques): ok
- Calculating pi with Leibniz: ok
- Towers of Hanoi: ok (recursive)
- Eight queens: ok
- Dijkstra's algorithm (single target): ok
- Genetic algorithm + traveling salesman
- Genetic algorithm + photo mosaic
- K-means clustering + shoulder k-choice
- Neural networks from scratch
- Tic-tac-toe (minimax)
- Connect-Four (minimax + alpha/beta pruning)
- MCMCMC estimator (Metropolis-coupled Markov Chain Monte-Carlo)
- Quantum BC (quantum betweenness centrality): unfinished

## Quantum Graph Analysis Project
This project demonstrates a conceptual approach to applying quantum computing for graph analysis, with a focus on exploring and measuring path properties in a network. Utilizing IBM's Qiskit framework, the project outlines the foundational steps required to encode a graph into a quantum state, perform quantum walks, amplify shortest paths, and measure these paths to infer properties such as betweenness centrality.

### Overview
Graph theory problems, such as finding the shortest paths and calculating betweenness centrality, are fundamental in various fields, including network analysis, optimization, and social network analysis. Quantum computing offers promising avenues to tackle these problems, potentially surpassing classical computing limitations in scalability and efficiency.

This project is divided into several conceptual components:

- Graph State Preparation: Encoding a graph's structure into a quantum state.
- Quantum Walks: Simulating quantum walks on the graph to explore paths.
- Amplification of Shortest Paths: Using quantum amplitude amplification to increase the likelihood of observing shortest paths.
- Path Measurement: Collapsing the quantum state to measure and analyze the paths within the graph.

### Requirements
- Python 3.7+
- Qiskit
- NetworkX (for graph manipulation and representation)
- Matplotlib (for visualization)
### Installation
Ensure you have Python installed on your system. Install the required packages using pip:

{pip install qiskit networkx matplotlib}

### Usage
The project is structured into modular functions, each corresponding to a step in the quantum graph analysis process. Due to the conceptual nature of the project, these functions serve as a foundation and require further development for practical applications.

1. Graph State Preparation: Encode your graph into a quantum circuit.
2. Quantum Walks: Simulate quantum walks on the encoded graph.
3. Amplify Shortest Paths: Apply amplitude amplification tailored to highlight shortest paths.
4. Measure Paths: Measure the quantum circuit to obtain path information.

### Contributing
This project is open-source and welcomes contributions. Whether you are looking to fix bugs, enhance the functionality, or extend the project's scope, your input is appreciated.

1. Fork the repository.
2. Create a new branch for your feature (git checkout -b feature/AmazingFeature).
3. Commit your changes (git commit -m 'Add some AmazingFeature').
3. Push to the branch (git push origin feature/AmazingFeature).
4. Open a Pull Request.

### License
Distributed under the MIT License. See LICENSE for more information.

### Acknowledgments
IBM Qiskit Team for the quantum computing framework.
The Python community for the excellent libraries that made this project possible.