# An Agent-Based Model to Study Stack Overflow Interactions

### Project description

Social Q\&A communities are social media platforms that focus on user knowledge exchange and have become an important platform for users to ask questions, seek information, and share knowledge.

In this project we used an ABM approach to model question-answer upvoting interactions among users on Stack Overflow, which allows us to explore the underlying upvoting feedback mechanisms that result in the emergent behavior that the number of upvotes and reputation distribution follow a power law.

### Content

This repository contains:
* A [report](https://github.com/AaronDC60/ABM_stackoverflow_network/blob/main/ABM_stackoverflow.pdf) of the project
* The [slides](https://github.com/AaronDC60/ABM_stackoverflow_network/blob/main/presentation.pdf) of a presentation that was given about the project (in an unfinished stage)
* A [code](https://github.com/AaronDC60/ABM_stackoverflow_network/tree/main/code) folder with all the source code and the code to generate the results
* A folder with the results of the [sensitivity analysis](https://github.com/AaronDC60/ABM_stackoverflow_network/tree/main/sensitivity_analysis)
* A folder with all the [figures](https://github.com/AaronDC60/ABM_stackoverflow_network/tree/main/figures) from the report

### Code

The source code of this ABM is spread over three files:
* [`user.py`](https://github.com/AaronDC60/ABM_stackoverflow_network/blob/main/code/user.py), which contains the implementation of the different entities (user, question and answer).
* [`model.py`](https://github.com/AaronDC60/ABM_stackoverflow_network/blob/main/code/model.py), which contains the implementation that represents the overall network.
* [`utils.py`](https://github.com/AaronDC60/ABM_stackoverflow_network/blob/main/code/utils.py), which contains several helper functions.

The following dependencies are required to run the model:
* [scikit-learn](https://scikit-learn.org/stable/install.html)
* [numpy](https://numpy.org/install/)
* [scipy](https://scipy.org/install/)
* (unittest)

In the code folder, there is a [test file](https://github.com/AaronDC60/ABM_stackoverflow_network/blob/main/code/tests/test_model.py) that can be used to see if the model runs properly.
This test file can be ran from the terminal using the following commands:
```
cd code
python -m tests.test_model
```

Other files that are located in the code folder are:
* [`tags.txt`](https://github.com/AaronDC60/ABM_stackoverflow_network/blob/main/code/tags.txt), which contains the relative frequency of the 12 most popular Stack Overflow tags.
* [`local_sensitivity_analysis.py`](https://github.com/AaronDC60/ABM_stackoverflow_network/blob/main/code/local_sensitivity_analysis.py), which contains the code used to generate the data for the LSA.
* [`global_sensitivity_analysis.py`](https://github.com/AaronDC60/ABM_stackoverflow_network/blob/main/code/global_sensitivity_analysis.py), which contains the code used to generate the data for the GSA.
* [`results.ipynb`](https://github.com/AaronDC60/ABM_stackoverflow_network/blob/main/code/results.ipynb), which contains the code used to generate all the results.

To run the sensitivity analysis, there are few additional dependencies required:
* [SALib](https://salib.readthedocs.io/en/latest/getting-started.html#installing-salib)
* pandas
* multiprocessing
* matplotlib (visualization of results)

### Sensitivity analysis

The output of the sensitivity analysis is written to csv and these files are stored within the [data](https://github.com/AaronDC60/ABM_stackoverflow_network/tree/main/sensitivity_analysis/data) folder.
Inside this folder there are notebooks containing the code that produced the visualization of the SA

* [`lsa.ipynb`](https://github.com/AaronDC60/ABM_stackoverflow_network/blob/main/sensitivity_analysis/lsa.ipynb)
* [`gsa.ipynb`](https://github.com/AaronDC60/ABM_stackoverflow_network/blob/main/sensitivity_analysis/gsa.ipynb)
