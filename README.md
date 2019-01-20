# Running FbProphet on Spark using Python

[![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#contributors)

In this example repository, I have provided a sample input.csv file in the `./data` directory for you to use. The input.csv looks like the following:

| total    | timestamp           | metric_type        | app   |
| :------: | :-----------------: | :----------------: | :---: |
| 184240.0 | 2018-06-21 00:00:00 | total_transactions | app_0 |
| 189187.0 | 2018-06-21 00:05:00 | total_transactions | app_0 |
| 180939.0 | 2018-06-21 00:10:00 | total_transactions | app_0 |
| 176656.0 | 2018-06-21 00:15:00 | total_transactions | app_0 |
| 175127.0 | 2018-06-21 00:20:00 | total_transactions | app_0 |
| 184399.0 | 2018-06-21 00:25:00 | total_transactions | app_0 |
| 178659.0 | 2018-06-21 00:30:00 | total_transactions | app_0 |
| 174519.0 | 2018-06-21 00:35:00 | total_transactions | app_0 |
| 180096.0 | 2018-06-21 00:40:00 | total_transactions | app_0 |

There are a total of 4 different metric_types:

- total_transactions
- total_cpu
- total_resptime
- total_memory

There are 31 unique apps.

There is 1 month of data for each app-metric combination. The data timestamps are from `2018-06-21 00:00:00` to `2018-07-21 23:55:00`.

This code will run FBProphet on the input.csv dataset for each app-metric combination so that we can predict the next days values for each application and their individual respective metric_types.

Running on a 4 core i7, 16 gb ram laptop:

| Description of Run   | Number of effective fits | Total Time |
| :------------------: | :----------------------: | :--------: |
| One app all metrics  | 4                        | 13 minutes |
| All apps all metrics | 124                      | 31 minutes |

## Installation

**Note:** this code was written using python3. It can still work with python2, but some editing may be required.

FbProphet

The wonderful folks at facebook have provided awesome documentation on the installation process for fbprophet depending on your OS on their [github project page](https://facebook.github.io/prophet/docs/installation.html)

For all other dependencies:

Create and source virtualenv for project

Mac

```bash
python -m virtualenv venv
source ./venv/bin/activate
```

Windows

```cmd
python -m virtualenv venv
.\venv\Scripts\activate.bat
```

- Note, you may need to install virtualenv first into global pip via `$ pip install virtualenv`

Install via Pip

```bash
pip install -r requirements.txt
```

## Running the Application

Once you have sourced your virtualenv you have access to the spark-submit command

```bash
spark-submit sparkprophet.py
```

## Optional Additional Steps

Using this as a template for running fbprophet on your data is a good start, but in order to maxmize your results you would need to perform a grid search to find the optimal input parameters to the fbprophet algorithm. This can also be done via spark by creating a second grid dataframe with your parameters and all possible combinations and applying a crossjoin on the input dataset. Then using the groupby to run the algorithm over each app-metric-parametercombo combination. Finally you would need to have a reduceby key step to find the grid that produced the minimum mse score to use as your best fit parameters for the run. In the future, I will try and write up a more explicit tutorial on how to do a grid search using apache spark.

Lastly, this code is intended to run in spark standalone (local) mode. It can easily be modified to run on a spark cluster, but I figured the tutorial would be more clear if I just show running in local mode. Maybe I will include these steps in the tutorial writeup I mentioned previously.

## Contributors

Thanks goes to these wonderful people ([emoji key](https://github.com/kentcdodds/all-contributors#emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore -->
| [<img src="https://avatars3.githubusercontent.com/u/9095499?v=4" width="100px;"/><br /><sub><b>Andrew Sidlo</b></sub>](https://github.com/asidlo)<br />[🤔](#ideas-asidlo "Ideas, Planning, & Feedback") [💻](https://github.com/asidlo/sparkprophet/commits?author=asidlo "Code") [🎨](#design-asidlo "Design") [📖](https://github.com/asidlo/sparkprophet/commits?author=asidlo "Documentation") | [<img src="https://avatars1.githubusercontent.com/u/33138515?v=4" width="100px;"/><br /><sub><b>Devarsh Raghnathbhai Patel</b></sub>](https://github.com/Devarsh-UTD)<br />[🤔](#ideas-Devarsh-UTD "Ideas, Planning, & Feedback") [💻](https://github.com/asidlo/sparkprophet/commits?author=Devarsh-UTD "Code") | [<img src="https://avatars3.githubusercontent.com/u/4262190?v=4" width="100px;"/><br /><sub><b>Rohit Chauhan</b></sub>](http://www.topmist.com)<br />[🤔](#ideas-Saarus "Ideas, Planning, & Feedback") |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/kentcdodds/all-contributors) specification. Contributions of any kind welcome!