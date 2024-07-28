# Overview
This is an auto-bidding training framework to help participants implement and evaluate their bidding strategies. This framework includes three modules: data processing, strategy training, and offline evaluation. Several industry-proven baseline strategies, such as reinforcement learning-based bidding and online linear programming-based bidding, are included in the framework. Participants can utilize this framework to develop a well-trained auto-bidding strategy based on the training dataset. Since the auction system cannot be used for evaluation during offline training, participants can rely on the provided framework for a basic offline assessment to ensure the code implementation meets the competition requirements.

. 


## Dependencies
```
conda create -n nips-bidding-env python=3.9.12 pip=23.0.1
conda activate nips-bidding-env
pip install -r requirements.txt
```

# Usage
## Dataset Link
Due to the large size of the data file, it has been split into multiple parts for download.

https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/general_track_data_period_7-8.zip

https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/general_track_data_period_9-10.zip

https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/general_track_data_period_11-12.zip

https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/share/general_track_data_period_13.zip




## Data Processing
Download the traffic granularity data and place it in the biddingTrainENv/data/ folder.
The directory structure under data should be:
```
NeurIPS_Auto_Bidding_General_Track_Baseline
|── data
    |── traffic
        |── period-7.csv
        |── period-8.csv
        |── period-9.csv
        |── period-10.csv
        |── period-11.csv
        |── period-12.csv
        |── period-13.csv
```

Run this script to convert the traffic granularity data into trajectory data required for RL training, facilitating subsequent RL policy training.
```
python  bidding_train_env/dataloader/rl_data_generator.py
```

## strategy training
### reinforcement learning-based bidding

#### IQL(Implicit Q-learning) Model
Load the training data and train the IQL bidding strategy.
```
python main/main_iql.py 
```
Use the IqlBiddingStrategy as the PlayerBiddingStrategy for evaluation.
```
bidding_train_env/strategy/__init__.py
from .iql_bidding_strategy import IqlBiddingStrategy as PlayerBiddingStrategy
```
#### BC(behavior cloning) Model
Load the training data and train the BC bidding strategy.
```
python main/main_bc.py 
```
Use the BcBiddingStrategy as the PlayerBiddingStrategy for evaluation.
```
bidding_train_env/strategy/__init__.py
from .bc_bidding_strategy import BcBiddingStrategy as PlayerBiddingStrategy
```

### online linear programming-based bidding
#### OnlineLp Model
Load the training data and train the OnlineLp bidding strategy.
```
python main/main_onlineLp.py 
```
Use the OnlineLpBiddingStrategy as the PlayerBiddingStrategy for evaluation.
```
bidding_train_env/strategy/__init__.py
# from .onlinelp_bidding_strategy import OnlineLpBiddingStrategy as PlayerBiddingStrategy
```

## offline evaluation
Load the training data to construct an offline evaluation environment for assessing the bidding strategy offline.
```
python main/main_test.py
```

# Appendix

## Traffic granularity data format
The training dataset is derived from advertising delivery data generated via the auction system where multiple advertisers compete against each other. Participants can use this dataset to recreate the historical delivery process of all advertisers across all impression opportunities. The training dataset includes 7 delivery periods. Each delivery period contains approximately 500,000 impression opportunities and is divided into 48 steps. There are 48 advertisers competing for these opportunities. The dataset consists of approximately 170 million records, with a total size of 25G. The specific data format is as follows:

* **(c1) deliveryPeriodIndex**: Represents the index of the current delivery period.
* **(c2) advertiserNumber**: Represents the unique identifier of the advertiser.
* **(c3) advertiserCategoryIndex**: Represents the index of the advertiser's industry category.
* **(c4) budget**: Represents the advertiser's budget for a delivery period.
* **(c5) CPAConstraint**: Represents the CPA constraint of the advertiser.
* **(c6) timeStepIndex**: Represents the index of the current decision time step.
* **(c7) remainingBudget**: Represents the advertiser's remaining budget before the current step.
* **(c8) pvIndex**: Represents the index of the impression opportunity.
* **(c9) pValue**: Represents the conversion action probability when the advertisement is exposed to the customer.
* **(c10) pValueSigma**: Represents the prediction probability uncertainty.
* **(c11) bid**: Represents the advertiser's bid for the impression opportunity.
* **(c12) xi**: Represents the winning status of the advertiser for the impression opportunity, where 1 implies winning the opportunity and 0 suggests not winning the opportunity.
* **(c13) adSlot**: Represents the won ad slot. The value ranges from 1 to 3, with 0 indicating not winning the opportunity .
* **(c14) cost**: Represents the cost that the advertiser needs to pay if the ad is exposed to the customer.
* **(c15) isExposed**: Represents whether the ad in the slot was displayed to the customer, where 1 implies the ad is exposed and 0 suggests not exposed.
* **(c16) conversionAction**: Represents whether the conversion action has occurred, where 1 implies the occurrence of the conversion action and 0 suggests that it has not occurred.
* **(c17) leastWinningCost**: Represents the minimum cost to win the impression opportunity,i.e., the 4-th highest bid of the impression opportunity.
* **(c18) isEnd**: Represents the completion status of the advertising period, where 1 implies either the final decision step of the delivery period or the advertiser's remaining budget falling below the system-set minimum remaining budget.


## Training data example
### example-1

| c1 |  c2 | c3 |   c4   |  c5  | c6 |   c7   |  c8   |   c9    |  c10   |  c11  | c12 | c13 |  c14  | c15 | c16 |  c17  | c18 |
|----|-----|----|--------|------|----|--------|-------|---------|--------|-------|-----|-----|-------|-----|-----|-------|-----|
|  1 |  31 |  2 | 6500.00| 27.00|  5 | 5962.49| 101000| 0.0103542| 0.0021549 | 0.2845 |  1  |  1  | 0.2702|  1  |  0  | 0.1832|  0  |
|  1 |  22 |  6 | 7000.00| 38.00|  5 | 5988.25| 101000| 0.0070297| 0.0005213 | 0.2702 |  1  |  2  | 0.2154|  1  |  1  | 0.1832|  0  |
|  1 |  15 |  7 | 7000.00| 42.00|  5 | 6132.52| 101000| 0.0051392| 0.0004312 | 0.2154 |  1  |  3  | 0.1832|  0  |  0  | 0.1832|  0  |
|  1 |  39 |  3 | 6000.00| 30.00|  5 | 5443.27| 101000| 0.0062134| 0.0007254 | 0.1832 |  0  |  0  | 0     |  0  |  0  | 0.1832|  0  |
|  1 |  43 |  9 | 7500.00| 25.00|  5 | 6421.81| 101000| 0.0045392| 0.0006215 | 0.1099 |  0  |  0  | 0     |  0  |  0  | 0.1832|  0  |

This example presents an impression opportunity involving the top five advertisers. The top three advertisers, numbered 31, 22, and 15, won the impression opportunity with the highest bids and were allocated to ad slots 1, 2, and 3, respectively. During this impression, slots 1 and 2 were exposed to the customer, while slot 3 remained unexposed. Consequently, ads in slots 1 and 2 need to pay 0.2702 and 0.2154, respectively. Additionally, the customer engaged in a conversion action with the ad in slot 2.


### example-2

| c1 | c2 | c3 | c4     | c5   | c6 | c7     | c8   | c9        | c10       | c11   | c12 | c13 | c14   | c15 | c16 | c17    | c18 |
|----|----|----|--------|------|----|--------|------|-----------|-----------|-------|-----|-----|-------|-----|-----|--------|-----|
| 3  | 48 | 6  | 7500.00| 40.00| 1  | 7500.00| 1    | 0.0032157 | 0.0003567 | 0.1345| 0   | 0   | 0     | 0   | 0   | 0.1628 | 0   |
| 3  | 48 | 6  | 7500.00| 40.00| 1  | 7500.00| 2    | 0.0146256 | 0.0021352 | 0.5852| 0   | 0   | 0     | 0   | 0   | 0.6421 | 0   |
| 3  | 48 | 6  | 7500.00| 40.00| 1  | 7500.00| 3    | 0.0054324 | 0.0007631 | 0.1924| 1   | 1   | 0.1673 | 1   | 1   | 0.1454 | 0   |
| 3  | 48 | 6  | 7500.00| 40.00| 1  | 7500.00| 4    | 0.0073145 | 0.0006529 | 0.2786| 0   | 0   | 0     | 0   | 0   | 0.2862 | 0   |
| …  |
| 3  | 48 | 6  | 7500.00| 40.00| 2  | 7341.25| 20901| 0.0076453 | 0.0006579 | 0.2856| 0   | 0   | 0     | 0   | 0   | 0.3245 | 0   |
| 3  | 48 | 6  | 7500.00| 40.00| 2  | 7341.25| 20902| 0.0139234 | 0.0012358 | 0.5629| 1   | 2   | 0     | 0   | 0   | 0.4782 | 0   |
| 3  | 48 | 6  | 7500.00| 40.00| 2  | 7341.25| 20903| 0.0077212 | 0.0006579 | 0.3045| 0   | 0   | 0     | 0   | 0   | 0.3122 | 0   |
| 3  | 48 | 6  | 7500.00| 40.00| 2  | 7341.25| 20904| 0.0021341 | 0.0001873 | 0.0926| 0   | 0   | 0     | 0   | 0   | 0.1151 | 0   |
| …  |
| 3  | 48 | 6  | 7500.00| 40.00| 43 | 0.00   | 895201| 0.0065274 | 0.0005689 | 0.0000| 0   | 0   | 0     | 0   | 0   | 0.1243 | 1   |
| 3  | 48 | 6  | 7500.00| 40.00| 43 | 0.00   | 895202| 0.0032125 | 0.0002986 | 0.0000| 0   | 0   | 0     | 0   | 0   | 0.2986 | 1   |
| 3  | 48 | 6  | 7500.00| 40.00| 43 | 0.00   | 895203| 0.0112986 | 0.0013253 | 0.0000| 0   | 0   | 0     | 0   | 0   | 0.0932 | 1   |
| 3  | 48 | 6  | 7500.00| 40.00| 43 | 0.00   | 895204| 0.0051678 | 0.0006782 | 0.0000| 0   | 0   | 0     | 0   | 0   | 0.1687 | 1   |

This example presents a data sample illustrating an advertiser's bidding process across time steps within a delivery period. The advertiser has a budget of 7500, a CPA constraint of 40, and belongs to industry category 6. Throughout different time steps, the advertiser engages in bidding for every available impression and obtains the corresponding results. During this period, the advertiser's remaining budget decreases correspondingly. Additionally, the advertiser adjusts their bidding strategy based on prior performance, although this adjustment will not be directly evident in the data.



   