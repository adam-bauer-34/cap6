# The Climate Asset Pricing model -- AR6 (CAP6)

Written by: Adam Michael Bauer

Affiliation: Department of Physics, University of Illinois at Urbana-Champaign, Loomis Laboratory, 1110 West Green St, Urbana, IL 61801, USA

To contact: adammb4 [at] illinois [dot] edu

## Contents
1. [Code description](#codedesc)
2. [Publication](#pubs)
3. [Model](#model)
4. [Directory overview](#directover)
   1. [main](#main)
   2. [analysis notebook](#analnote)
   3. [src](#src)
      1. [`climate.py`](#climate)
      2. [`cost.py`](#cost)
      3. [`damage_simulation.py`](#damsim)
      4. [`damage.py`](#dam)
      5. [`emit_baseline.py`](#emitbase)
      6. [`optimization.py`](#opt)
      7. [`storage_tree.py`](#storetree)
      8. [`tools.py`](#tool)
      9.  [`tree.py`](#tree)
      10. [`utility.py`](#util)
      11. [analysis](#anal)
          1. [`climate_output.py`](#outclimate)
          2. [`risk_decomp.py`](#riskdecomp)
          3. [`tree_diagram.py`](#treediagram)
          4. [`output_unpacker.py`](#outputunpack)
   4. [data](#data)
1. [Copyright statement](#dontgetgot)

## Code description <a name=“codedesc”></a>
This code is an updgraded, modified, and refactored version of [EZClimate](https://github.com/Litterman/EZClimate).  It features an updated climate model based off of the *transient climate respone to emissions* (TCRE), updated damage functions and cost curves based on estimates made from the IPCC’s sixth assessment report, a refactored and more accurate damage interpolation scheme, new flexible emissions baselines based off of the shared socioeconomic pathways (SSPs), and new Jupyter notebooks making model output easily analyzable by the user. The code is also thoroughly commented, thus making it accessible for users wishing to modify it and use it in their own studies.

## Publication <a name=“pubs”></a>
We are preparing a publication which uses this code to analyze the controlling factors on CO<sub>2</sub> price paths. We will post a link and abstract when the publication is prepared and submitted.

## Model <a name=“model”></a>
TCREZClimate is a dynamical asset pricing model designed to calculate the optimal price path of CO<sub>2</sub> emissions. The code recursively solves for the economic utility, assuming Epstein-Zin preferences, within a path dependent binomial tree framework. See our paper for a full description of the model.

## Directory overview <a name=“directover”></a>

### `BPW_main.py` <a name=“main”></a>
The main file for the model. 
To run: python BPW_main.py.

### Analysis_notebook <a name=“analnote”></a>
The analysis notebook for the model. See the notebook for directions on use. 

### src <a name=“src”></a>

#### `climate.py` <a name=“climate”></a>
Contains the `BPWClimate` class, which contains all the relevant methods and parameters associated with our climate emulator. Two main quantities are calculated in this class: the temperature as a function of time, and the CO<sub>2</sub> concentrations in the atmosphere given an emissions pathay.

#### `cost.py` <a name=“cost”></a>
Contains `BPWCost` class, which calculates the total cost ot society of CO<sub>2</sub> emissions, as well as the price of mitigation.

#### `damage_simulation.py` <a name=“damsim”></a>
Contains the `DamageSimulation` class, which using Monte Carlo creates a set of potential damage pathways. This, in essense, defies the landscape of fragility in the model, which impacts representitive agent decisions to abate or not abate CO<sub>2</sub> emissions.

#### `damage.py` <a name=“dam”></a>
Contains the `BPWDamage` class, which calculates climate damages at each node in the tree.

#### `emit_baseline.py` <a name=“emitbase”></a>
Contains the `BPWEmissionBaseline` class, which dictates the emissions baseline for the model run; choices include each of the SSPs, 1—5. They are extended to 2400 using an interpolation procedure, see the code for details

#### `optimization.py` <a name=“opt”></a>
Contains two classes: `GeneticAlgorithm` and `GradientDescent`, which are used to find the optimal utility. The genetic algorithm is a stochastic optimizaiton routine which can find optima of an objective function without requiring its derivative. The output of the genetic algorithm is used as the *initial guess* of the gradient descent algorithm, which further refines the optimal economic utility.

#### `storage_tree.py` <a name=“storetree”></a>
Contains three classes: `BigStorageTree`, `SmallStorageTree` and `BaseStorageTree`; `BigStorageTree` and `SmallStorageTree` are subclasses of `BaseStorageTree`. These classes are used to store values of various model quantites during optimization, such as the economic utility, consumption, and price.

#### `tools.py` <a name=“tool”></a>
A set of miscellaneous functions used throughout the model, such as various I/O functions and integration calculators.

#### `tree.py` <a name=“tree”></a>
Contains the `TreeModel` class, which defines the binomial tree underlying the optimization calculation.

#### `utility.py` <a name=“util”></a>
Contains `EZUtility` class, which calculates the utility throughout our model.

#### analysis <a name=“anal”></a>

##### `climate_output.py` <a name=“outclimate”></a>
Contains `ClimateOutput` object, which is used to calculate various model quantites after optimization has been run to be saved for analysis.

##### `risk_decomp.py` <a name=“riskdecomp”></a>
Contains `RiskDecomposition` object, which carries out a risk decomposition routine to parse the influence of various economic parameters on the cost of carbon.

##### `tree_diagrams.py` <a name=“treediagram”></a>
Contains `TreeDiagram` object, which can be used in the analysis notebook to create tree diagrams to visualize the evolution of various model outputs within the binomial tree structure.

##### `output_unpacker.py` <a name=“outputunpack”></a>
Contains `OutputUnpacker` object, which is used by the master analysis notebook to “unpack” the model output and make a set of interesting quantities for analysis.

### data
This directory will contain the output of the model. It currently contains two files: `SSP_baselines.csv` and `BPW_research_runs.csv`. The first dictates the emissions in each SSP for the years 2020 — 2100. The second contains the sets of model paramters for our model runs in our paper. 

## Copyright statement <a name=“dontgetgot”></a>
MIT License

Copyright (c) 2022 Adam Michael Bauer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


