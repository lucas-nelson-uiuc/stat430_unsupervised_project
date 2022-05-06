import pandas as pd
import numpy as np

from sklearn.manifold import TSNE

from statsbombpy import sb

import matplotlib.pyplot as plt
import seaborn as sns


def gather_sb_competitions():
    # gather all competitions provided in OpenAccess
    all_comps = sb.competitions()
    
    # catch-all for later merger
    comps = []
    seasons = []
    num_matches = []

    # populate lists with competition, season, number-of-matches pairs
    for idx, season_info in all_comps[['competition_id', 'season_id', 'competition_name']].iterrows():
        try:
            temp_df = sb.matches(competition_id=season_info['competition_id'], season_id=season_info['season_id'])
            comps.append(all_comps.loc[idx, 'competition_name'])
            seasons.append(all_comps.loc[idx, 'season_name'])
            num_matches.append(temp_df.shape[0])
        except:
            comps.append(all_comps.loc[idx, 'competition_name'])
            seasons.append(all_comps.loc[idx, 'season_name'])
            num_matches.append(0)

    # update all_comps to include number of matches in given season
    updated_comps = pd.merge(
        all_comps,
        pd.DataFrame({
        'competition_name':comps,
        'season_name':seasons,
        'num_matches':num_matches
        }),
        on=['competition_name', 'season_name']
    )

    # output number of matches per country-gender-competition-season pair
    return updated_comps.groupby(
        ['country_name', 'competition_gender', 'competition_name', 'season_name']
        ).agg({'num_matches':'mean'}).astype('int')

list_to_string = lambda x: ','.join([str(i) for i in x])
off_cols = [
    'player', 'location', 'type',
    'pass_angle', 'pass_outcome', 'pass_end_location', 'pass_cross', 'pass_goal_assist',
    'pass_shot_assist', 'pass_through_ball', 'pass_technique',
    'shot_outcome', 'shot_statsbomb_xg', 'shot_technique', 'shot_type',
    'dribble_outcome'
    ]
def preprocessing_events_df(events_df, o_cols=off_cols, o_attrs=['Pass', 'Shot', 'Dribble', 'Cross'], time_cols=['match_id', 'minute', 'second']):
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

    list_to_string = lambda x: ','.join([str(i) for i in x])
    
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

def plt_tsne_plot(df, perps, rss, hues=[]):
    f, axs = plt.subplots(nrows=1, ncols=len(hues), figsize=(16,9))
    for i, col in enumerate(hues):
        sns.scatterplot(x='proj_x', y='proj_y', hue=col, data=df, ax=axs[i])
    return f

def plt_tsne_subplots(df, n_rows=None, n_cols=None, perps=[5,10,20,30,40,50], rss=[95, 433], drop_cols=[]):
    '''
    Create n_rows x n_cols grid of subplots for t-SNE plots of
    passed (subset) dataframe and perplexity/random
    state values
    '''
    
    n_rows, n_cols = len(rss), len(perps)
    f, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16,9))
    
    for i, rs in enumerate(rss):
        for j, perp in enumerate(perps):
            tsne = TSNE(n_components=2, perplexity=perp, random_state=rs)
            tsne_fit = tsne.fit_transform(df.drop(columns=drop_cols))
            tsne_data = pd.DataFrame(tsne_fit, columns=['x_proj', 'y_proj'])
            
            sns.scatterplot(x='x_proj', y='y_proj', data=tsne_data, ax=axs[i % n_rows, j % n_cols], legend=False)
            
            axs[i % n_rows, j % n_cols].set_title(f'perp={perp}, rs={rs}')
            axs[i % n_rows, j % n_cols].set_xticks([])
            axs[i % n_rows, j % n_cols].set_yticks([])
            axs[i % n_rows, j % n_cols].set_xlabel('')
            axs[i % n_rows, j % n_cols].set_ylabel('')

    return f

def gen_joint_grid(pitch):
    return pitch.jointgrid(
        figheight=10,  # the figure is 10 inches high
        left=None,  # joint grid center-aligned
        bottom=0.075,  # grid starts 7.5% in from the bottom of the figure
        marginal=0.1,  # marginal axes heights are 10% of grid height
        space=0,  # 0% of the grid height reserved for space between axes
        grid_width=0.9,  # the grid width takes up 90% of the figure width
        title_height=0,  # plot without a title axes
        axis=False,  # turn off title/ endnote/ marginal axes
        endnote_height=0,  # plot without an endnote axes
        grid_height=0.8)

def plt_actions_on_pitch(master_df, pitch):
    
    pass_df = master_df[master_df['type'] == 'Pass'].copy()
    pass_df[['location_x', 'location_y']] = pass_df[['location_x', 'location_y']].astype('float64')
    dribble_df = master_df[master_df['type'] == 'Dribble'].copy()
    dribble_df[['location_x', 'location_y']] = dribble_df[['location_x', 'location_y']].astype('float64')
    cross_df = master_df[master_df['type'] == 'Cross'].copy()
    cross_df[['location_x', 'location_y']] = cross_df[['location_x', 'location_y']].astype('float64')
    shot_df = master_df[master_df['type'] == 'Shot'].copy()
    shot_df[['location_x', 'location_y']] = shot_df[['location_x', 'location_y']].astype('float64')

    f1, axs1 = gen_joint_grid(pitch)
    pitch.scatter(pass_df['location_x'], pass_df['location_y'],
        c='#800020', alpha=0.7, ec='black', ax=axs1['pitch'])
    sns.histplot(y=pass_df['location_y'], ax=axs1['left'], element='step', color='#ba495c')
    sns.histplot(x=pass_df['location_x'], ax=axs1['top'], element='step', color='#ba495c')

    f2, axs2 = gen_joint_grid(pitch)
    pitch.scatter(cross_df['location_x'], cross_df['location_y'],
        c='#800020', alpha=0.7, ec='black', ax=axs2['pitch'])
    sns.histplot(y=cross_df['location_y'], ax=axs2['left'], element='step', color='#ba495c')
    sns.histplot(x=cross_df['location_x'], ax=axs2['top'], element='step', color='#ba495c')

    f3, axs3 = gen_joint_grid(pitch)
    pitch.scatter(dribble_df['location_x'], dribble_df['location_y'],
        c='#800020', alpha=0.7, ec='black', ax=axs3['pitch'])
    sns.histplot(y=dribble_df['location_y'], ax=axs3['left'], element='step', color='#ba495c')
    sns.histplot(x=dribble_df['location_x'], ax=axs3['top'], element='step', color='#ba495c')

    f4, axs4 = gen_joint_grid(pitch)
    pitch.scatter(shot_df['location_x'], shot_df['location_y'],
        c='#800020', alpha=0.7, ec='black', ax=axs4['pitch'])
    sns.histplot(y=shot_df['location_y'], ax=axs4['left'], element='step', color='#ba495c')
    sns.histplot(x=shot_df['location_x'], ax=axs4['top'], element='step', color='#ba495c')

def plt_actions_distribtuion(master_df, total_minutes_df):
    
    f, axs = plt.subplots(nrows=1, ncols=2, sharey=True)
    sns_df = master_df.groupby(['player', 'type']).agg({'location_x':'count'}).rename(columns={'location_x':'count'}).reset_index()
    sns_upgrade_df = pd.merge(
        sns_df,
        total_minutes_df.reset_index('match_id', drop=True).groupby('player').sum().reset_index(),
        on='player'
    )
    sns_upgrade_df['per_90'] = sns_upgrade_df['count'] * 90 / sns_upgrade_df['minutes']

    sns.barplot(y='player', x='minutes', color='#ba495c', data=sns_upgrade_df.sort_values('minutes', ascending=False), ax=axs[0])
    sns.barplot(y='player', x='per_90', hue='type', palette='colorblind',
    data=sns_upgrade_df.sort_values('minutes', ascending=False), ax=axs[1])

    axs[0].set_title('Total Minutes Played')
    axs[1].set_title('On-ball Actions per 90 Minutes')
    axs[0].set_xlabel('Minutes')
    axs[1].set_xlabel('Count')
    axs[0].set_ylabel('Player')
    axs[1].set_ylabel('')
    axs[0].legend([])
    axs[1].legend(bbox_to_anchor=(1,0.8), frameon=False)
    
    plt.show()
