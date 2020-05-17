from riotwatcher import LolWatcher, ApiError
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import networkx as nx
import os

# Global variables
API_KEY = os.environ.get('API_KEY')
REGION = 'euw1'
QUEUE = 'RANKED_SOLO_5x5'
watcher = LolWatcher(API_KEY)
N_MATCHES = 5
N_MATCHES_OTHERS = 5
N_PLAYERS = 5
COLOR_SEQ = px.colors.qualitative.D3
TIER = RANK = role = None

# Get the latest version of the game from data dragon
versions = watcher.data_dragon.versions_for_region(REGION)
champions_version = versions['n']['champion']
current_champ_list = watcher.data_dragon.champions(champions_version)
champ_dict = {}
for key in current_champ_list['data']:
    row = current_champ_list['data'][key]
    champ_dict[row['key']] = row['id']
    
# Functions
def get_tier_and_rank(summonerId):
    try:
        ranked_stats = watcher.league.by_summoner(REGION, summonerId)
    except ApiError as err:
        if err.response.status_code == 429:
            return 429
        else:
            raise
    for queue in ranked_stats:
        if queue['queueType'] == QUEUE:
            return queue['tier'], queue['rank']
    return None, None

def get_role(role, lane):
    if lane != 'BOTTOM':
        return lane
    else:
        return 'ADC' if role == 'DUO_CARRY' else 'SUPPORT'

def create_network_graph(players_list=[]):
    # Creating the graph
    network_graph = nx.Graph()
    network_graph.add_edges_from(players_list)
    pos = nx.layout.spring_layout(network_graph)

    # Creating the edges
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5,color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    for edge in network_graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    # Creating the nodes
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[node for node in network_graph.nodes()],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            colorscale='YlGnBu',
            color=[],
            size=50,
            line=dict(width=2)
        )
    )
    for node in network_graph.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    # Adding color and hovertext
    summoners_list = [player[0] for player in players_list]
    champions_list = [player[1] for player in players_list]
    for node in node_trace['text']:
        if node == summoners_list[0]:
            node_trace['marker']['color'] += tuple(['#3B81C8'])
        elif node in summoners_list:
            node_trace['marker']['color'] += tuple(['#FF8000'])
        elif node in champions_list:
            node_trace['marker']['color'] += tuple(['#3BC893'])
        else:
            print('Error: ' + node + ' is not a summoner nor a champion')
            raise
    
    # Creating the visualization
    network_fig = go.Figure(data=[edge_trace, node_trace],
                         layout=go.Layout(
                            title='<br>Champions used',
                            titlefont=dict(size=16),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)),
    )
    return network_fig

# Get summoner statistics
def get_summoner_stats(summoner_name):
    global TIER
    global RANK
    global role
    
    try:
        summoner = watcher.summoner.by_name(REGION, summoner_name)
    except ApiError as err:
        if err.response.status_code == 429:
            return 429
        elif err.response.status_code == 404:
            return None
        else:
            raise

    # Get player stats
    TIER, RANK = get_tier_and_rank(summoner['id'])
    if None in [TIER, RANK]:
        return None
    try:
        matches = watcher.match.matchlist_by_account(REGION, summoner['accountId'])
    except ApiError as err:
        if err.response.status_code == 429:
            return 429
        else:
            raise
    
    last_match = matches['matches'][0]
    role = get_role(last_match['role'], last_match['lane'])

    gameIds = []
    for match in matches['matches']:
        if get_role(match['role'], match['lane']) == role:
            gameIds.append((match['gameId'], match['champion']))

    network_edges = []

    stats = []
    for gameId, championId in gameIds[:N_MATCHES]:
        match_stats = {}
        match_stats['champion'] = champ_dict[str(championId)]
        try:
            match_detail = watcher.match.by_id(REGION, gameId)
        except ApiError as err:
            if err.response.status_code == 429:
                return 429
            else:
                raise
        duration = match_detail['gameDuration'] / 60
        totalDamage = {match_detail['teams'][0]['teamId']: 0, match_detail['teams'][1]['teamId']: 0}
        for participant in match_detail['participants']:
            if participant['championId'] == championId:
                teamId = participant['teamId']
                K = participant['stats']['kills']
                D = participant['stats']['deaths']
                A = participant['stats']['assists']
                KDA = (K + A) / D if D > 0 else K + A
                
                match_stats['win'] = participant['stats']['win']
                match_stats['K'] = K
                match_stats['D'] = D
                match_stats['A'] = A
                match_stats['KDA'] = KDA
                match_stats['gold'] = participant['stats']['goldEarned']
                match_stats['goldpm'] = participant['stats']['goldEarned'] / duration
                match_stats['cspm'] = participant['stats']['totalMinionsKilled'] / duration
                match_stats['damage'] = participant['stats']['totalDamageDealtToChampions']
                match_stats['vision'] = participant['stats']['visionScore']
            totalDamage[participant['teamId']] += participant['stats']['totalDamageDealtToChampions']
        match_stats['damage%'] = match_stats['damage'] / totalDamage[teamId]
        stats.append(match_stats)
        network_edges.append((summoner_name, match_stats['champion']))
    df_player = pd.DataFrame(stats)
    df_mean_player = df_player.mean()
    K, D, A = df_mean_player['K'], df_mean_player['D'], df_mean_player['A']
    KDA = (K + A) / D if D > 0 else K + A
    player_win = df_mean_player['win'] * 100
    player_KDA = KDA
    player_vision = df_mean_player['vision']
    player_goldpm = df_mean_player['goldpm']
    player_cspm = df_mean_player['cspm']
    player_damage = df_mean_player['damage%'] * 100

    # Get similar players stats
    stats = []
    try:
        entries = watcher.league.entries(REGION, QUEUE, TIER, RANK)
    except ApiError as err:
        if err.response.status_code == 429:
            return 429
        else:
            raise
    summonerNames = [entry['summonerName'] for entry in entries]

    for summonerName in summonerNames:
        if summonerName == summoner_name:
            continue
        
        try:
            summonerId = watcher.summoner.by_name(REGION, summonerName)['accountId']
            matches = watcher.match.matchlist_by_account(REGION, summonerId)
        except ApiError as err:
            if err.response.status_code == 429:
                return 429
            else:
                raise
        
        gameIds = []
        for match in matches['matches']:
            if get_role(match['role'], match['lane']) == role:
                gameIds.append((match['gameId'], match['champion']))
            if len(gameIds) == N_MATCHES_OTHERS:
                break
        
        if len(gameIds) < N_MATCHES_OTHERS:
            continue
        
        wins = K = D = A = KDA = goldpm = cspm = vision = damage_perc = 0
        for gameId, championId in gameIds:
            try:
                match_detail = watcher.match.by_id(REGION, gameId)
            except ApiError as err:
                if err.response.status_code == 429:
                    return 429
                else:
                    raise
            duration = match_detail['gameDuration'] / 60
            totalDamage = {match_detail['teams'][0]['teamId']: 0, match_detail['teams'][1]['teamId']: 0}
            for participant in match_detail['participants']:
                if participant['championId'] == championId:
                    teamId = participant['teamId']
                    current_K = participant['stats']['kills']
                    current_D = participant['stats']['deaths']
                    current_A = participant['stats']['assists']
                    current_KDA = (current_K + current_A) / current_D if current_D > 0 else current_K + current_A

                    wins += participant['stats']['win']
                    K += current_K
                    D += current_D
                    A += current_A
                    KDA += current_KDA
                    goldpm += participant['stats']['goldEarned'] / duration
                    cspm += participant['stats']['totalMinionsKilled'] / duration
                    vision += participant['stats']['visionScore']
                    current_damage = participant['stats']['totalDamageDealtToChampions']
                    network_edges.append(('Summoner{}'.format(len(stats)+1), champ_dict[str(championId)]))
                totalDamage[participant['teamId']] += participant['stats']['totalDamageDealtToChampions']
            damage_perc += current_damage / totalDamage[teamId]
        playerStats = {}
        playerStats['win_rate'] = wins / N_MATCHES_OTHERS * 100
        playerStats['global_KDA'] = (K + A) / D if D > 0 else K + A
        playerStats['KDA_mean'] = KDA / N_MATCHES_OTHERS
        playerStats['goldpm_mean'] = goldpm / N_MATCHES_OTHERS
        playerStats['cspm_mean'] = cspm / N_MATCHES_OTHERS
        playerStats['vision_mean'] = vision / N_MATCHES_OTHERS
        playerStats['damage%_mean'] = damage_perc / N_MATCHES_OTHERS * 100
        stats.append(playerStats)
        
        if len(stats) == N_PLAYERS:
            break
        
    df_others = pd.DataFrame(stats)
    df_mean_others = df_others.mean()
    similar_win = df_mean_others['win_rate']
    similar_KDA = df_mean_others['global_KDA']
    similar_vision = df_mean_others['vision_mean']
    similar_goldpm = df_mean_others['goldpm_mean']
    similar_cspm = df_mean_others['cspm_mean']
    similar_damage = df_mean_others['damage%_mean']
    
    stats_names = ['win rate', 'KDA', 'vision',
                   'gold per minute', 'creeps per minute', 'damage dealt ratio']

    max_stats = [100,
                 max(player_KDA, similar_KDA),
                 max(player_vision, similar_vision),
                 max(player_goldpm, similar_goldpm),
                 max(player_cspm, similar_cspm),
                 max(player_damage, similar_damage)]

    player_stats = [player_win, player_KDA, player_vision,
                    player_goldpm, player_cspm, player_damage]

    similar_stats = [similar_win, similar_KDA, similar_vision,
                     similar_goldpm, similar_cspm, similar_damage]

    df_radar_chart = pd.DataFrame(dict(r=[player_stats[i] / max_stats[i] for i in range(len(max_stats))] +
                        [similar_stats[i] / max_stats[i] for i in range(len(max_stats))],
                     theta=stats_names * 2,
                     player=[summoner_name] * 6 + ['similar players'] * 6))
    radar_chart = px.line_polar(data_frame=df_radar_chart, r='r', theta='theta', color='player',
                       color_discrete_sequence=COLOR_SEQ,
                       line_close=True, title=summoner_name + " vs similar players statistics")
    radar_chart.update_traces(fill='toself')
    
    df = pd.DataFrame({'player': [summoner_name, 'similar players'],
                       'win rate': [player_win, similar_win],
                       'KDA': [player_KDA, similar_KDA],
                       'vision': [player_vision, similar_vision],
                       'gold per minute': [player_goldpm, similar_goldpm],
                       'creeps per minute': [player_cspm, similar_cspm],
                       'damage dealt ratio': [player_damage, similar_damage]})
    
    win_rate = px.bar(df, x='win rate', y='player', orientation='h', labels={'player':''},
                       color='player', color_discrete_sequence=COLOR_SEQ)
    win_rate.update_layout(yaxis=dict(showticklabels=False), showlegend=False)
    
    KDA = px.bar(df, x='KDA', y='player', orientation='h', labels={'player':''},
                       color='player', color_discrete_sequence=COLOR_SEQ)
    KDA.update_layout(yaxis=dict(showticklabels=False), showlegend=False)
    
    vision = px.bar(df, x='vision', y='player', orientation='h', labels={'player':''},
                       color='player', color_discrete_sequence=COLOR_SEQ)
    vision.update_layout(yaxis=dict(showticklabels=False), showlegend=False)
    
    goldpm = px.bar(df, x='gold per minute', y='player', orientation='h', labels={'player':''},
                       color='player', color_discrete_sequence=COLOR_SEQ)
    goldpm.update_layout(yaxis=dict(showticklabels=False), showlegend=False)
    
    cspm = px.bar(df, x='creeps per minute', y='player', orientation='h', labels={'player':''},
                       color='player', color_discrete_sequence=COLOR_SEQ)
    cspm.update_layout(yaxis=dict(showticklabels=False), showlegend=False)
    
    damage = px.bar(df, x='damage dealt ratio', y='player', orientation='h', labels={'player':''},
                       color='player', color_discrete_sequence=COLOR_SEQ)
    damage.update_layout(yaxis=dict(showticklabels=False), showlegend=False)
    
    network_graph = create_network_graph(network_edges)
    
    return (radar_chart, win_rate, KDA, vision, goldpm, cspm, damage, network_graph)

stats_names = ['win rate', 'KDA', 'vision',
               'gold per minute', 'creeps per minute', 'damage dealt ratio']

df_radar_chart = pd.DataFrame(dict(r=[0] * 12,
                 theta=stats_names * 2))
radar_chart_empty = px.line_polar(df_radar_chart, r='r', theta='theta',
                   title="Summoner vs similar players statistics")
radar_chart_empty.update_traces(fill='toself')

df_bar_charts = pd.DataFrame({'player': ['summoner', 'similar players'],
                       'win rate': [0, 0],
                       'KDA': [0, 0],
                       'vision': [0, 0],
                       'gold per minute': [0, 0],
                       'creeps per minute': [0, 0],
                       'damage dealt ratio': [0, 0]})
                       
win_rate_empty = px.bar(df_bar_charts, x='win rate', y='player', orientation='h', labels={'player':''})
win_rate_empty.update_layout(yaxis=dict(showticklabels=False), showlegend=False)

KDA_empty = px.bar(df_bar_charts, x='KDA', y='player', orientation='h', labels={'player':''})
KDA_empty.update_layout(yaxis=dict(showticklabels=False), showlegend=False)

vision_empty = px.bar(df_bar_charts, x='vision', y='player', orientation='h', labels={'player':''})
vision_empty.update_layout(yaxis=dict(showticklabels=False), showlegend=False)

goldpm_empty = px.bar(df_bar_charts, x='gold per minute', y='player', orientation='h', labels={'player':''})
goldpm_empty.update_layout(yaxis=dict(showticklabels=False), showlegend=False)

cspm_empty = px.bar(df_bar_charts, x='creeps per minute', y='player', orientation='h', labels={'player':''})
cspm_empty.update_layout(yaxis=dict(showticklabels=False), showlegend=False)

damage_empty = px.bar(df_bar_charts, x='damage dealt ratio', y='player', orientation='h', labels={'player':''})
damage_empty.update_layout(yaxis=dict(showticklabels=False), showlegend=False)

network_fig_empty = create_network_graph()



# Dash
app = dash.Dash(__name__)
server = app.server

colors = {
    'text': '#7FDBFF'
}

markdown_text = '''
League of Legends Dashboard tool developed by Laura Almón Manzano for a University project.
Done with a Riot's personal API, which only supports 20 requests per second or 100 every 2 minutes.
To use it, enter a summoner's name and wait for the graphs to update. Please, be patient!
You can use mine: "Láudano". Hope you like it!
'''

app.layout = html.Div([
    html.H1(
        children='League of Legends Dashboard',
        style={'textAlign': 'center'}#, 'color': colors['text']}
    ),
    dcc.Markdown(
        children=markdown_text
    ),
    html.Div([
        html.Div(
            children='Summoner\'s name:'
        ),
        dcc.Input(
            id='summoner-name-input', type='search', placeholder='Summoner\'s name', debounce=True
        ),
        html.Div(
            id='summoner-name-error-output'
        )]
    ),
    dcc.Loading(
        id='loading',
        type='default',
        children=html.Div(id='loading-output')
    ),
    html.Div([
        html.H2(
            id='summoner-name-output',
            children='',
            style={'textAlign': 'center'}#, 'color': colors['text']}
        ),
        html.Div([
            html.Div(
                [html.P('TIER'), html.H6(id='tier_text')],
                id='tier'
            ),
            html.Div(
                [html.P('RANK'), html.H6(id='rank_text')],
                id='rank'
            ),
            html.Div(
                [html.P('ROLE'), html.H6(id='role_text')],
                id='role'
            )],
            style={'columnCount': 3}
        ),
        html.Div([
            dcc.Graph(
                id='player-vs-similar-stats',
                figure=radar_chart_empty
            ),
            html.Div([
                dcc.Graph(
                    id='win-rate',
                    figure=win_rate_empty
                ),
                dcc.Graph(
                    id='goldpm',
                    figure=goldpm_empty
                ),
                dcc.Graph(
                    id='KDA',
                    figure=KDA_empty
                ),
                dcc.Graph(
                    id='cspm',
                    figure=cspm_empty
                ),
                dcc.Graph(
                    id='vision',
                    figure=vision_empty
                ),
                dcc.Graph(
                    id='damage',
                    figure=damage_empty
                )],
                style={'columnCount': 3}
            ),
            html.Div(
                dcc.Graph(id='network', figure=network_fig_empty)
            )]
        )]
    )]
)

# Select summoner name
@app.callback(
    [Output('loading-output', 'children'),
    Output('summoner-name-output', 'children'),
    Output('summoner-name-error-output', 'children'),
    Output('tier_text', 'children'),
    Output('rank_text', 'children'),
    Output('role_text', 'children'),
    Output('player-vs-similar-stats', 'figure'),
    Output('win-rate', 'figure'),
    Output('goldpm', 'figure'),
    Output('KDA', 'figure'),
    Output('cspm', 'figure'),
    Output('vision', 'figure'),
    Output('damage', 'figure'),
    Output('network', 'figure')],
    [Input('summoner-name-input', 'value')]
)
def update_summoner_name(value):
    summoner_name = value
    if summoner_name in ['', None]:
        return ['', '', '', '', '', '', radar_chart_empty,
            win_rate_empty, KDA_empty, vision_empty, goldpm_empty, cspm_empty, damage_empty,
            network_fig_empty]
    summoner_stats = get_summoner_stats(summoner_name)
    if summoner_stats is None:
        return ['', '', 'Summoner name not found.', '', '', '', radar_chart_empty,
            win_rate_empty, KDA_empty, vision_empty, goldpm_empty, cspm_empty, damage_empty,
            network_fig_empty]
    elif summoner_stats == 429:
        return ['', '', 'Too many requests. Try again in a few seconds.', '', '', '', radar_chart_empty,
            win_rate_empty, KDA_empty, vision_empty, goldpm_empty, cspm_empty, damage_empty,
            network_fig_empty]
    radar_chart, win_rate, KDA, vision, goldpm, cspm, damage, network_fig = summoner_stats
    if None in [radar_chart, win_rate, KDA, vision, goldpm, cspm, damage]:
        return ['', '', 'Error. Try again in a few seconds.', '', '', '', radar_chart_empty,
            win_rate_empty, KDA_empty, vision_empty, goldpm_empty, cspm_empty, damage_empty,
            network_fig_empty]
    return ['', summoner_name + ' statistics', '', TIER, RANK, role, radar_chart,
        win_rate, KDA, vision, goldpm, cspm, damage, network_fig]

if __name__ == '__main__':
    app.run_server(debug=True)