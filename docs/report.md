Introduction
============

Motivation
----------

The application of machine learning methods to predict dynamical systems
is increasing sharply. By their design, machine learning models offer
tremendous flexibility, capable of capturing complex behaviour and
patterns, while maintaining relatively tiny computational footprint (at
least once learned).

Modern wind turbines have multiple sensors installed and provide
constant data stream outputs; however, there are some important
quantities where installing physical sensors is either impractical or
the sensor technology is not sufficiently advanced. In such cases, real
sensors may be too expensive or too noisy in real conditions to be used
to provide measurements.

This is where “virtual sensors” come in. It’s a matter of defining a
model in the sense of machine learning, which will enable new signals
(virtual measurements) to be created on the basis of those we know or
can measure.

Problem description
-------------------

### Reference Turbine : IEA 15MW WIND TURBINE MODEL

Reference turbines are open-source designs of a complete wind power
system, including simulation and design models. In particular, they can
be used to evaluate the performance and cost of proposed modifications
prior to prototype development. NREL (USA) has released its latest 15MW
reference wind turbine.

The **IEA 15MW wind turbine** design is available for use by the wind
energy community in input files that support a variety of analysis
tools, including OpenFAST, HAWC2, WISDEM and HawtOpt2. In addition, the
data represented in the graphs and tables of the [report
](https://www.nrel.gov/docs/fy20osti/75698.pdf) detailing the wind
turbine specifications are also available in
[Excel](https://github.com/IEAWindTask37/IEA-15-240-RWT/tree/master/Documentation)
files.

This wind turbine is illustrated on this <a href='#fig:turbine:illustration'>figure</a>. The
figure also shows the positions of the sensors we have access to on the
turbine, and the positions of the sensors for which we want to predict
the output using machine learning. As inputs, we then have some
displacements or rotation measurements from whose bending moments need
to be predicted as output.

<a id='fig:turbine:illustration'>![IEA 15MW wind turbine and
sensors](image/IEA15MW_with_sensors.jpg)</a>


### Wind Energy Virtual Sensors Benchmark <a id="dtu:benchmark"></a>

A crucial part of machine learning is to have access to a large amount
of high-quality data. Additionally, the data needs to be labelled in
order to perform supervised learning. This means we need a substantial
dataset that gives us access to both input and output data to enable our
algorithms to learn the dynamics our model needs to reproduce (our
*surrogate model*).

The IEA 15MW wind turbine gives access to OpenFast simulator. As a
result, it will be possible to define a synthetic response of the wind
turbine to different wind conditions.

In 2023, Nicolay Dimitrow, from DTU Wind and Energy Systems initiated a
benchmark which provides an open benchmarking dataset that can be used
for trying different virtual sensing approaches as well as for any other
study on loads and condition monitoring.

This means that an experimental design has been defined to sample
different operating points of the wind turbine by defining a certain
number of parameters.

A 7-D sample space with 6 environmental variables and one operational
(yaw misalignment) was sampled to provide environmental condition bounds
for load simulations. As an illustration this <a href='#fig:doe:pairplot'> figure</a>
illustrates how experimental space is sampled in the benchmark design.

<a id='fig:doe:pairplot'>![7-D sample space](image/DoE_7D_pairplot.png)</a>

Datasets
===============

As mentioned, DTU has provided a set of outputs supplied by `OpenFast`
using the wind data defined in paragraph <a href="#dtu:benchmark">benchmark</a>. One file can
be retrieved per experiment (one experiment per set of design
parameters). We therefore have 1000 files. These files are available
[here](https://www.zenodo.org/communities/wind-vs-benchmark/).

As output, each file provides 147 time series sampled at 100Hz over
700s. This gives us access to the wind speed at the center of the rotor,
as well as to certain accelerations measured at certain points on the
wind turbine, and also to forces and moments. These data, not normally
available in reality, will enable us to perform supervised learning. The
list of 147 signals names supplied by DTU is given [here](./columns.txt).

Sampling at 100Hz is too fast for the study we’re planning. We therefore
resampled the signals to 20Hz, first passing a low-pass filter to avoid
spectrum aliasing. In addition, to obtain signals that correspond to a
stationary regime, we removed the first 100 seconds, keeping only the
remaining 10 minutes (equivalent to the SCADA data sampling step).

This gives us an initial dataset of 1,000 experiments. In other words,
1000 files in parquet format. Each column in this file corresponds to
the last 10 minutes sampled at 20Hz from one of the 147 outputs provided
by OpenFast (i.e. 12,000 points per signal).


<a id="datastructure">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Time [s]</th>
      <th>Rotor azimuth [deg]</th>
      <th>Rotor rotational speed [rpm]</th>
      <th>Blade 1 pitch angle [deg]</th>
      <th>...</th>
      <th>Free wind speed Vy pos  -55.00,   0.00, -95.00</th>
      <th>Free wind speed Vz pos  -55.00,   0.00, -95.00</th>
      <th>Water surface elevation [m]</th>
    </tr>
</thead>
  <tbody>
    <tr>
      <th>0.00</th>
      <td>66.556067</td>
      <td>7.672028</td>
      <td>0.520026</td>
      <td>...</td>
      <td>11.832829</td>
      <td>0.599726</td>
      <td>-0.058321</td>
      </tr>
    <tr>
      <th>0.05</th>
      <td>68.860026</td>
      <td>7.677732</td>
      <td>0.528227</td>
      <td>...</td>
      <td>11.840616</td>
      <td>0.539415</td>
      <td>-0.098596</td>
      </tr>
    <tr>
      <th>0.10</th>
      <td>71.166235</td>
      <td>7.679375</td>
      <td>0.536139</td>
      <td>...</td>
      <td>11.845533</td>
      <td>0.438913</td>
      <td>-0.142723</td>
    </tr>
    <tr>
      <th>0.15</th>
      <td>73.470194</td>
      <td>7.677345</td>
      <td>0.543834</td>
      <td>...</td>
      <td>11.833485</td>
      <td>0.328324</td>
      <td>-0.188766</td>
      </tr>
    <tr>
      <th>0.20</th>
      <td>75.767403</td>
      <td>7.673720</td>
      <td>0.551354</td>
      <td>...</td>
      <td>11.810534</td>
      <td>0.216735</td>
      <td>-0.237811</td>
    </tr>
      <tr>
      <th>up to 12000 rows...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>-...</td>
    </tr>
  </tbody>
</table>
</a>

The data structure for one of these 1000 samples is given <a href="#datastructure">fig</a> for the first `id=Exp0`. All 147 signals are sampled each 50ms during 10 minutes.


Input signals
-------------

For the prediction we use 8 signals at the inputs of networks. Theses
signals are as follows:

-   **Tower top fore-aft acceleration ay [m/s2]:** forward-backward
    acceleration measured at the top of the tower in meters per second.

-   **Tower top side-side acceleration ax [m/s2]:** lateral acceleration
    measured at the top of the tower in meters per second.

-   **Tower mid for-aft acceleration ay [m/s2]:** forward-backward
    acceleration measured at the mid-height of wind turbine tower,
    indicating the movement of the central part of the tower in wind
    direction.

-   **Tower mid side-side acceleration ax [m/s2]:** lateral acceleration
    measured at the mid-height of wind turbine tower, indicating the
    lateral movement of the central part of the tower.

-   **Tower top rotation x [deg]:** this measure represents the rotation
    around the horizontal axis at the top of the tower, expressed in
    degrees.

-   **Tower top rotation y [deg]:** this measure represents the rotation
    around the vertical axis at the top of the tower, expressed in
    degrees.

-   **Tower mid rotation x [deg]:** this measure represents the rotation
    around the horizontal axis at mid-height of the tower, expressed in
    degrees.

-   **Tower mid rotation y [deg]:** this measure represents the rotation
    around the vertical axis at mid-height of the tower, expressed in
    degrees.

<a id="tab:input">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Time [s]</th>
      <th>Tower top fore-aft acceleration ay [m/s2]</th>
      <th>Tower top side-side acceleration ax [m/s2]</th>
      <th>Tower mid fore-aft acceleration ay [m/s2]</th>
      <th>Tower mid side-side acceleration ax [m/s2]</th>
      <th>Tower top rotation x [deg]</th>
      <th>Tower top rotation y [deg]</th>
      <th>Tower mid rotation x [deg]</th>
      <th>Tower mid rotation y [deg]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.00</th>
      <td>-0.089793</td>
      <td>0.292771</td>
      <td>-0.451854</td>
      <td>-0.271650</td>
      <td>0.46575</td>
      <td>-0.221151</td>
      <td>0.421875</td>
      <td>-0.070203</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>-0.029697</td>
      <td>0.171801</td>
      <td>-0.566831</td>
      <td>-0.179218</td>
      <td>0.46125</td>
      <td>-0.217698</td>
      <td>0.421875</td>
      <td>-0.069329</td>
    </tr>
    <tr>
      <th>0.10</th>
      <td>0.013529</td>
      <td>0.040707</td>
      <td>-0.563472</td>
      <td>-0.054931</td>
      <td>0.46125</td>
      <td>-0.213114</td>
      <td>0.421875</td>
      <td>-0.068388</td>
    </tr>
      <tr>
      <th>up to 12000 rows...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
  </tbody>
</table>
</a>

Target signals
--------------

From the 8 inputs signals presented in the previous section we will
predict 6 signals. The different signals. These signals are as follows:

-   **Mudline moment Mx[KNm]:** This is the moment around the axis X of
    mudline, expressed in kilonewton-meter.

-   **Mudline moment My[KNm]:** It’s the moment around the axis Y of
    mudline expressed in kilonewton-meter.

-   **Mudline moment Mz[KNm]:** This is the moment around the axis Z of
    mudline expressed in kilonewton-meter.

-   **Waterline moment Mx[KNm]:** This is the moment around the axis X
    of waterline, expressed in kilonewton-meter.

-   **Waterline moment My[KNm]:** This is the moment around the axis Y
    of waterline, expressed in kilonewton-meter.

-   **Waterline moment Mz[KNm]:** This is the moment around the axis Z
    of waterline, expressed in kilonewton-meter.

<a id="tab:target">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Time [s]</th>
      <th>Mudline moment Mx[kNm]</th>
      <th>Mudline moment My[kNm]</th>
      <th>Mudline moment Mz[kNm]</th>
      <th>Waterline moment Mx[kNm]</th>
      <th>Waterline moment My[kNm]</th>
      <th>Waterline moment Mz[kNm]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.00</th>
      <td>122300.083752</td>
      <td>11632.042227</td>
      <td>7655.904384</td>
      <td>264567.690504</td>
      <td>31845.901568</td>
      <td>7621.016674</td>
    </tr>
    <tr>
      <th>0.05</th>
      <td>123523.029784</td>
      <td>11946.845293</td>
      <td>7401.308883</td>
      <td>264999.483360</td>
      <td>32185.395920</td>
      <td>7372.876025</td>
    </tr>
    <tr>
      <th>0.10</th>
      <td>123776.701048</td>
      <td>12655.058163</td>
      <td>6810.098925</td>
      <td>264169.798740</td>
      <td>32897.378960</td>
      <td>6827.471046</td>
    </tr>
      <tr>
      <th>up to 12000 rows...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
  </tbody>
</table>
</a>

Normalization
--------------------------------------
Let's take a look at the statistical description of input and target signals


### Input Statistics
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tower top fore-aft acceleration ay [m/s2]</th>
      <th>Tower top side-side acceleration ax [m/s2]</th>
      <th>Tower mid fore-aft acceleration ay [m/s2]</th>
      <th>Tower mid side-side acceleration ax [m/s2]</th>
      <th>Tower top rotation x [deg]</th>
      <th>Tower top rotation y [deg]</th>
      <th>Tower mid rotation x [deg]</th>
      <th>Tower mid rotation y [deg]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>0.002137</td>
      <td>-0.000770</td>
      <td>0.000450</td>
      <td>-0.000307</td>
      <td>0.390212</td>
      <td>-0.184182</td>
      <td>0.373969</td>
      <td>-0.052710</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.134382</td>
      <td>0.139669</td>
      <td>0.290012</td>
      <td>0.162966</td>
      <td>0.064641</td>
      <td>0.014259</td>
      <td>0.033061</td>
      <td>0.008070</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.331137</td>
      <td>-0.528690</td>
      <td>-0.861611</td>
      <td>-0.543853</td>
      <td>0.230625</td>
      <td>-0.224915</td>
      <td>0.294750</td>
      <td>-0.075000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.392174</td>
      <td>0.359047</td>
      <td>0.763519</td>
      <td>0.484980</td>
      <td>0.528750</td>
      <td>-0.153350</td>
      <td>0.433125</td>
      <td>-0.034805</td>
    </tr>
  </tbody>
</table>


### Target Statistics
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mudline moment Mx[kNm]</th>
      <th>Mudline moment My[kNm]</th>
      <th>Mudline moment Mz[kNm]</th>
      <th>Waterline moment Mx[kNm]</th>
      <th>Waterline moment My[kNm]</th>
      <th>Waterline moment Mz[kNm]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>103474.724264</td>
      <td>9534.993099</td>
      <td>99.795798</td>
      <td>232941.854360</td>
      <td>24580.950652</td>
      <td>99.781762</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9115.613425</td>
      <td>2567.477798</td>
      <td>3566.809877</td>
      <td>19625.576972</td>
      <td>5260.768656</td>
      <td>3556.660551</td>
    </tr>
    <tr>
      <th>min</th>
      <td>77001.755600</td>
      <td>3357.899366</td>
      <td>-12870.842835</td>
      <td>184043.675664</td>
      <td>12052.194208</td>
      <td>-12810.351074</td>
    </tr>
    <tr>
      <th>max</th>
      <td>123776.701048</td>
      <td>16756.273888</td>
      <td>8550.295853</td>
      <td>271114.169796</td>
      <td>39416.365136</td>
      <td>8518.574790</td>
    </tr>
  </tbody>
</table>

It can be noticed that the mean values are all different; some values are
close to 0 and others exceed 10000. The signals are of different scales,
which may be a problem for the neural network, so it is necessary to
normalize them. 

#### MinMax normalization

The __minmax__ normalization method consists in applying the
following formula on each signal:

$$
x_{ij}^{(normalize)}=\frac{x_{ij}-x_{j}^{(min)}}{x_{j}^{(max)}-x_{j}^{(min)}} 
$$

with $x_{ij}$ the i-th value of the j-th signal, $x_{j}^{(max)}$ the
highest value of the j-th signal and $x_{j}^{(min)}$ the smallest value
of the j-th signal.

Dataset length
--------------

> If we make this new assumption - that output signals do not depend temporally on input signals over a 10-minute window, but over a  shorter time window - we can deduce from the initial dataset a different dataset, better suited to the definition of our surragate model.

Assuming that a window size of 1 minute is enough to capture the time dependancy of the surrogate model, a __learning dataset__ was then defined by reducing the size of input/output space of our signal to 1200 (instead of 12000). It means 1 minute timeserie instead of 10 minutes and it means also that we increase the number of samples available for learning by a factor 10. We then have more experiments and the datastructure length is then _up to 1200_ and not _up to 12000_ anymore.
<a href='#fig:learning:dataset'>Figure</a> illustrates the way __learning dataset__ is deduced from __original dataset__ 

<a id='fig:learning:dataset'><img src="./image/learning_dataset.svg" alt="image" width="900" height="auto"></a>


Dataset balance
---------------

Once our learning dataset is standardized according __min max__ normalization given statistical properties of all input/target signals, it is also randomly divided into training and testing dataset.
90% (9000/10000 samples) of the data will be used for training and the remaining 10%(1000/10000 samples) for
testing. 

To validate that the testing dataset sufficiently samples the space defined by the 7 parameters of the plan, wind speed in the hub will be extracted from the 147 signals provided by `OpenFast` simulations.
```python
wind_columns = [
    "Free wind speed Vx pos    0.00,   0.00,-150.00",
    "Free wind speed Vy pos    0.00,   0.00,-150.00",
    "Free wind speed Vz pos    0.00,   0.00,-150.00"
]

```

One of this 10000 signals is illustrated on this <a href='#fig:wind:profile'>figure</a> where the _Wind Speed_ against the _Y_ axis is given for 1 minutes.

<a id='fig:wind:profile'><img src="./image/free_wind_profile.png" ></a>

On each of these 10000 signals, we compute $\mu_{V_y}$ (Wind Mean) and $\sigma_{V_y}$ (Standard Deviation of the Wind). 
Then in $(\mu, \sigma)$ space each value is plotted on <a href='#fig:train:test:scatter'>figure</a>.

The observation of the figure reveals that the
test data are located in the training data region, thus emphasizing the
consistency between the way training and testing datasets are split. 
This approach makes it possible to
evaluate where (which range) the model is able to generalize , ie 
evaluate its performance on unpublished
data.

<a id='fig:train:test:scatter'><img src="./image/train_test_scatter.png" ></a>

See [Wind Process Notebook](../notebooks/wind_process.ipynb) for further explications

### Summary

At the end, what we want is to find a black box ( a surrogate model ) that, once it learned, can deduced the target signal given the input signals.


![Virtual Sensing](image/virtual_sensing_aim.png)


#### Shape considerations
It remains another degree of freedom. Given 8 $X$ inputs signals the surrogate model may deduce $Y$ output timeseries one by one or all in once.

* One By One

$ (10000, 8, 1200) \rightarrow (10000, 1, 1200)$ 

* All in Once

$ (10000, 8, 1200) \rightarrow (10000, 6, 1200)$ 


Neural Networks
===============

To establish connections between the input signals and the signals
targeted for prediction, various types of neural networks were employed
in our approach, including convolutional neural networks (CNNs) and
recurrent neural networks (RNNs). The utilization of these neural
network architectures facilitates the extraction and understanding of
complex patterns within the data, contributing to the effectiveness of
the predictive model.In the subsequent sections, we will delve into a
detailed presentation of each neural network employed in our study.

Recurrent Neural Network
------------------------

Recurrent neural networks (RNN) are a class of deep learning models
designed to process sequential data. Unlike traditional neural networks,
RNN store previous information and use it to influence the processing of
new data.

![Hidden state of a RNN memory cell](image/image6.png)

Long Short-Term Memory
----------------------

Long Short-Term Memory is a type of recurrent neural networks that was
created to bypass the vanishing gradient problem, a challenge that
arises during the training of traditional RNNs. A cell of LSTM network
is maindly composed to 3 parts:

-   **Forget Gate:** in this part of the cell, some information
    previously stored in memory is intentionally erased.This gate
    enables the LSTM to discard irrelevant or outdated information,
    allowing the network to focus on more relevant data.

-   **Input Gate:** This part of the LSTM cell is responsible for
    incorporating new information into the cell’s memory. The input gate
    regulates the flow of incoming information, determining which data
    is important to retain and add to the existing memory.

-   **Output Gate:** The output controls the flow of information that is
    passed on to the next layer of the neural network or that used as
    the final output.

![Exemple of LSTM Network](image/image10.png)

Convolutional Neural Network
----------------------------

Convolutional neural networks (CNNs) are a type of neural network
designed primarily for processing grid-structured data, such as images.
They use convolution layers to extract meaningful features, such as
contours and patterns, from input data. This approach allows CNNs to
capture local information while maintaining a certain hierarchy in the
extracted characteristics. Due to their ability to learn complex models
from spatial data, CNNs are commonly used in areas such as computer
vision for tasks such as image classification and object detection.

![Exemple of using CNN](image/image11.png)

U-Net
-----

U-Net is a neural network architecture used primarily for image
segmentation. Its U-shaped structure incorporates an encoder to extract
features and a decoder to reconstruct the segmented image. Direct
connections between layers preserve contextual information, making it a
popular choice for accurate image segmentation.

![Exemple of a U-net structure](image/image12.png)


Network structures
==================

After data processing, the next phase involves the development of
models. The initial step is to determine the loss function and
appropriate regression measurements. This involves selecting the error
function that the network will aim to minimize and specifying the
metrics used to evaluate and compare the performance of different
networks. Once this is done, the structure of different neural networks
and the hyperparameters of these networks must be chosen in the case of
the prediction of target signals one by one and in the case of the
prediction of six target signals together.

Loss function and Regression metrics
------------------------------------

The loss function that the different neural networks will try to
minimize is **the Means Square Error** (MSE). MSE is calculated by
taking the mean of the squares of the differences between the values
predicted by the model and the actual values. The mathematical formula
of the MSE for a dataset of size $n$ is as follows:

$$
MSE=\frac{\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^2}{n}
$$

Selection of Hyperparameters and functions
------------------------------------------

- utilisation de ray tune pour trouver le learning rate et le weight
decay - parler de General Parameters and functions ( ne pas oublier de
parler des samples)

Final Network Structures
------------------------

Cas de la prédiction des signaux( 1 à 1)\
- LSTM CNN CNN-LSTM U-Net\
Cas de la prédiction des 6 signaux ensembles\
- LSTM CNN CNN-LSTM U-net\

Results
=======

Conclusion
==========
