import pandas as pd
import numpy as np

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns



# convert list column to two string columns
def list_to_string(x):
    '''
    Convert list value to csv value
    '''
    return ','.join([str(i) for i in x])

def preprocessing_events_df(
    events_df,
    o_cols=[
        'player', 'location', 'type',
        'pass_outcome', 'pass_end_location', 'pass_cross', 'pass_goal_assist', 'pass_shot_assist', 'pass_through_ball', 'pass_technique',
        'shot_outcome', 'shot_statsbomb_xg', 'shot_technique', 'shot_type',
        'dribble_outcome'],
    o_attrs=['Pass', 'Shot', 'Dribble', 'Cross'],
    time_cols=['match_id', 'minute', 'second']
    ):
    '''
    Return dataframe that contains offense-related metrics
    found in `offensive_cols` and `offensive_attrs`

    > events_df: play-by-play dataframe of team formations,
                 match start/finish, and on-ball actions
    '''
    
    # if NAs are auto generated for me
    for col in o_cols:
        if col not in events_df.columns:
            o_cols.remove(col)

    # events from specific match with valid on-ball player data
    nonempty_df = events_df[(events_df['player_id'].notna()) & (events_df['location'].notna()) & (events_df['team'] == 'Arsenal')][o_cols + time_cols]

    # select specific offensive actions (types)
    nonempty_df = nonempty_df[nonempty_df['type'].isin(o_attrs)]

    # split x,y coordinates of location data
    nonempty_df = pd.merge(
        nonempty_df,
        nonempty_df['location'].apply(list_to_string).str.split(',', expand=True),
        left_index=True, right_index=True, how='outer'
        )
    nonempty_df.rename(columns={0:'location_x', 1:'location_y'}, inplace=True)

    # split x,y coordinates of passing event data
    nonempty_df = pd.merge(
        nonempty_df,
        nonempty_df[nonempty_df['type'] == 'Pass']['pass_end_location'].apply(list_to_string).str.split(',', expand=True),
        left_index=True, right_index=True, how='outer'
        )
    nonempty_df.rename(columns={0:'pass_end_x', 1:'pass_end_y'}, inplace=True)

    # update type column to include crosses
    nonempty_df['type'] = np.where(nonempty_df['pass_cross'] == 1, 'Cross', nonempty_df['type'])

    # return dataframe with desired events
    return nonempty_df.drop(columns=['location', 'pass_end_location'])


def gather_team_data(match_id, events_df):
    '''
    Return DataFrame of single metric values per
    match_id in a given season/competition pairing
    from StatsBomb OpenAccess database
    '''

    return pd.DataFrame(
        {match_id : {
            'xG' : events_df['shot_statsbomb_xg'].astype('float64').sum(),
            'shots' : events_df[events_df['type'] == 'Shot'].shape[0],
            'passes' : events_df[events_df['type'] == 'Pass'].shape[0],
            'dribbles' : events_df[events_df['type'] == 'Dribble'].shape[0],
            'goals' : events_df[events_df['shot_outcome'] == 'Goal'].shape[0]
        }}
    ).T

def gather_time_information(events_df):
    '''
    Return DataFrame of minute information recorded by players
    of specific team (e.g., Arsenal)
    '''

    return events_df[(events_df['player_id'].notna()) & (events_df['team'] == 'Arsenal')][['match_id', 'player', 'minute']]

def generate_tsne_subplots(df, perps=[5,10,20,30,40,50], rss=[95, 433], drop_cols=[]):
    '''
    Create 3x4 grid of subplots for t-SNE plots of
    passed (subset) dataframe and perplexity/random
    state values
    '''

    f, axs = plt.subplots(nrows=3, ncols=4, figsize=(16,9))
    for i, perp in enumerate(perps):
        for j, rs in enumerate(rss):
            tsne = TSNE(n_components=2, perplexity=perp, random_state=rs)
            tsne_fit = tsne.fit_transform(df.drop(columns=drop_cols))
            tsne_data = pd.DataFrame(tsne_fit, columns=['x_proj', 'y_proj'])
            
            sns.scatterplot(x='x_proj', y='y_proj', data=tsne_data, ax=axs[i//2, 2*(i%2) + j%2], legend=False)
            
            axs[i//2, 2*(i%2) + j%2].set_title(f'perp={perp}, rs={rs}')
            axs[i//2, 2*(i%2) + j%2].set_xticks([])
            axs[i//2, 2*(i%2) + j%2].set_yticks([])
            axs[i//2, 2*(i%2) + j%2].set_xlabel('')
            axs[i//2, 2*(i%2) + j%2].set_ylabel('')

    return f


