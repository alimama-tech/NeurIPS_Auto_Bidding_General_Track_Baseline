import os
import pandas as pd
import glob


class OnlineLp:
    '''
    OnlineLp model
    '''

    def __init__(self, dataPath):
        self.dataPath = dataPath

    def train(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        csv_files = glob.glob(os.path.join(self.dataPath, '*.csv'))
        print(csv_files)
        csv_files = sorted(csv_files)
        # select episode-0
        for i, csv_file_path in enumerate(csv_files[0:1]):
            print("开始分析" + csv_file_path)
            df = pd.read_csv(csv_file_path)
            episodeRes = self.onlinelp_for_specific_episode(df)
            episodeRes.to_csv(f'{save_path}/period.csv', index=False)
            print("完成分析" + csv_file_path)

    def onlinelp_for_specific_episode(self, df):
        df_filter = df[(df["pValue"] > 0) & (df["leastWinningCost"] > 0.0001)]
        grouped_df = df_filter.groupby('advertiserCategoryIndex')
        num_tick = 48
        max_budget = 6000
        interval = 10
        result_dfs = []

        for category, group in grouped_df:
            print("category构建开始:", category)
            sampled_group = group.sample(frac=1 / 8)
            sampled_group['realCPA'] = sampled_group['leastWinningCost'] / (sampled_group['pValue'] + 0.0001)
            for timestep in range(num_tick):
                timestep_filtered = sampled_group[sampled_group['timeStepIndex'] >= timestep]
                timestep_filtered = timestep_filtered.sort_values(by='realCPA').reset_index(drop=True)
                column_list = ["deliveryPeriodIndex", "advertiserCategoryIndex", "realCPA", "cum_cost"]
                timestep_filtered['cum_cost'] = timestep_filtered['leastWinningCost'].cumsum()
                timestep_filtered = timestep_filtered[column_list]
                timestep_filtered["timeStepIndex"] = timestep
                filtered_df = timestep_filtered[timestep_filtered['cum_cost'] < max_budget]
                last_selected = 0
                result = []

                for index, row in filtered_df.iterrows():
                    if row['cum_cost'] - last_selected >= interval:
                        result.append(row)
                        last_selected = row['cum_cost']
                    elif index == len(filtered_df) - 1:
                        result.append(row)
                        last_selected = row['cum_cost']

                final_df = pd.DataFrame(result)
                result_dfs.append(final_df)

        # 合并所有结果 DataFrames
        final_result_df = pd.concat(result_dfs).reset_index(drop=True)
        return final_result_df
