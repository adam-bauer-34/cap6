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
   2. [aux_notebooks](#auxnote)
   3. [src](#src)
   4. [data](#data)
   5. [notebooks] (#notes)
1. [Copyright statement](#dontgetgot)

## Code description <a name=“codedesc”></a>
This is the source code for the Climate Asset Pricing model -- AR6 (CAP6). CAP6 features an updated climate model based off of an effective transient climate respone to emissions (TCRE), updated damage functions and cost curves based on estimates made from the IPCC’s sixth assessment report, new flexible emissions baselines based off of the shared socioeconomic pathways (SSPs), and new Jupyter notebooks making model output easily analyzable by the user. The code is also thoroughly commented (we hope), thus making it accessible for users wishing to modify it and use it in their own studies.

## Publication <a name=“pubs”></a>
We are preparing a publication which uses this code and will post a link and abstract when the publication is prepared and submitted.

## Model <a name=“model”></a>
CAP6 is a dynamical asset pricing model designed to calculate the optimal price path of CO<sub>2</sub> emissions. The code recursively solves for the economic utility, assuming Epstein-Zin preferences, within a path dependent binomial tree framework. See our paper for a full description of the model.

## Directory overview <a name=“directover”></a>

### Main files <a name=“main”></a>
There are two "main" files. The first is `main.py`, which runs a single model run (or as many as you want). Which run you want to do is pulled from the `desired_runs` list in `main.py` (which, in turn, comes from `research_runs.csv` in the `data` directory).

The second "main" file is `main_ensemble.py`. This creates the "ensemble" model runs. **Warning**: this takes an insanely long time to run (order weeks) on a personal computer. We had a computing cluster at our disposal, which made this code managable (it still took 6-7 days).

### `aux_notebooks` <a name=“analnote”></a>
This directory contains the various notebooks we used to calibrate different model components.

### `src` <a name=“src”></a>
Contains the source code for the model. The "nuts and bolts", if you will.

### `data`
This directory will contain the output of the model when it is run. It also contains `research_runs.csv`, which tells the "main" files what the model parameters are for a given run. Additionally, it contains calibration data, such as the data points for different damage functions, emissions baselines, and so on.

### Notebooks <a name="notes"></a>
Each notebook in the top directory is an analysis notebook. These were used to create the figures in our publication. They come with varying levels of sloppiness. 

## Copyright statement <a name=“dontgetgot”></a>
MIT License

Copyright (c) 2023 Adam Michael Bauer

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


