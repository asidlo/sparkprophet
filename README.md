# Running FbProphet on Spark using Python

[![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#contributors)

In this example repository, I have provided a sample input.csv file in the 
`./data` directory for you to use. The input.csv looks like the following:

| timestamp           | metric   | app   | value               |
| :-----------------: | :------: | :---: | :-----------------: |
| 2019-01-01 00:00:00 | m1       | a     | 61.87483488182826   |
| 2019-01-01 00:05:00 | m1       | a     | 4.774629678532727   |
| 2019-01-01 00:10:00 | m1       | a     | 56.723598483827686  |
| 2019-01-01 00:15:00 | m1       | a     | 73.41004199189977   |
| 2019-01-01 00:20:00 | m1       | a     | 25.89179312049582   |
| 2019-01-01 00:25:00 | m1       | a     | 75.94699428222006   |
| 2019-01-01 00:30:00 | m1       | a     | 15.20946296181217   |
| 2019-01-01 00:35:00 | m1       | a     | 82.9956834656641    |
| 2019-01-01 00:40:00 | m1       | a     | 4.720798758063505   |


### Data Summary

- Unique metric_types: [m1, m2, m3]
- Unique apps: [a, b, c]

There is 3 months of data for each app-metric combination. The data timestamps 
are from `2019-01-01 00:00:00` to `2019-03-31 23:55:00`. This particular data is
generated as a random uniform distribution with values between 0, 100 exclusive.

This code will run FBProphet on the input.csv dataset for each app-metric 
combination so that we can predict the next days values for each application and
their individual respective metric_types.

Running on a 4 core i7, 16 gb ram laptop:

| Description of Run   | Number of effective fits | Total Time |
| :------------------: | :----------------------: | :--------: |
| One app all metrics  | 3                        | 33 seconds |
| All apps all metrics | 9                        | 56 seconds |

## Installation

This code was written and compiled using an anaconda3 environment. The required
packages are listed in the conda-requirements.txt and can be installed using the
following command: `conda create --name <env> --file <this file>`. This was test
using a macbook pro running Mojave 10.14.3. The respective pip requirements.txt
file is present as well, but you may need additional requirements to install
fbprophet / pystan. The instructions for installing those libs are located
[here](https://facebook.github.io/prophet/docs/installation.html).

As always, it is recommended to install all dependencies in a virtualenv of your
choosing.

Install via conda

```bash
conda create --name sparkprophet --file conda-requirements.txt
```

Install via Pip

```bash
pip install -r requirements.txt
```

## Running the Application

Once you have sourced your virtualenv you have access to the spark-submit command,
or you can run it like any other normal python script.

```bash
python sparkprophet.py
```

## Optional Additional Steps

Using this as a template for running fbprophet on your data is a good start, but
in order to maxmize your results you would need to perform a grid search to find
the optimal input parameters to the fbprophet algorithm. This can also be done 
via spark by creating a second grid dataframe with your parameters and all 
possible combinations and applying a crossjoin on the input dataset. Then using 
the groupby to run the algorithm over each app-metric-parametercombo combination.
Finally you would need to have a reduceby key step to find the grid that produced
the minimum mse score to use as your best fit parameters for the run.

Lastly, this code is intended to run in spark standalone (local) mode. It can 
easily be modified to run on a spark cluster, see the documentation on running
in [cluster-mode](https://spark.apache.org/docs/latest/cluster-overview.html).

## Contributors

Thanks goes to these wonderful people ([emoji key](https://github.com/kentcdodds/all-contributors#emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore -->
| [<img src="https://avatars3.githubusercontent.com/u/9095499?v=4" width="100px;" alt="Andrew Sidlo"/><br /><sub><b>Andrew Sidlo</b></sub>](https://github.com/asidlo)<br />[🤔](#ideas-asidlo "Ideas, Planning, & Feedback") [💻](https://github.com/asidlo/sparkprophet/commits?author=asidlo "Code") [🎨](#design-asidlo "Design") [📖](https://github.com/asidlo/sparkprophet/commits?author=asidlo "Documentation") | [<img src="https://avatars1.githubusercontent.com/u/33138515?v=4" width="100px;" alt="Devarsh Raghnathbhai Patel"/><br /><sub><b>Devarsh Raghnathbhai Patel</b></sub>](https://github.com/Devarsh-UTD)<br />[🤔](#ideas-Devarsh-UTD "Ideas, Planning, & Feedback") [💻](https://github.com/asidlo/sparkprophet/commits?author=Devarsh-UTD "Code") | [<img src="https://avatars3.githubusercontent.com/u/4262190?v=4" width="100px;" alt="Rohit Chauhan"/><br /><sub><b>Rohit Chauhan</b></sub>](http://www.topmist.com)<br />[🤔](#ideas-Saarus "Ideas, Planning, & Feedback") |
| :---:                                                                                                                                                                                                                                                                                                                                                                                                                  | :---:                                                                                                                                                                                                                                                                                                                                             | :---:                                                                                                                                                                                                                      |
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/kentcdodds/all-contributors) specification. Contributions of any kind welcome!