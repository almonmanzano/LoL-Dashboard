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
N_MATCHES = 10
N_MATCHES_OTHERS = 5
N_PLAYERS = 5
COLOR_SEQ = px.colors.qualitative.D3
TIER = RANK = role = None
similar_tab_content = []
older_tab_content = []
current_tab = 'similar'

# Get the latest version of the game from data dragon
#versions = watcher.data_dragon.versions_for_region(REGION)
#champions_version = versions['n']['champion']
#current_champ_list = watcher.data_dragon.champions(champions_version)
current_champ_list = watcher.data_dragon.champions('10.10.3208608')
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

def create_champs_network_graph(players_list=[]):
    # Creating the graph
    network_graph = nx.Graph()
    network_graph.add_edges_from(players_list)
    pos = nx.layout.spring_layout(network_graph, k=0.5)

    # Creating the edges
    edge_trace = go.Scatter(
        x=[], y=[],
        line=dict(color='#888'),
        hoverinfo='none', mode='lines'
    )
    for edge in network_graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    # Creating the nodes
    node_trace = go.Scatter(
        x=[], y=[],
        text=[node for node in network_graph.nodes()],
        hoverinfo='text', mode='markers+text',
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
                            margin=dict(b=50,l=25,r=25,t=50),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)),
    )
    return network_fig

def create_players_network_graph(summoner_name, players_list=[]):
    summoners_dict = {}
    for player1, player2 in players_list:
        if player1 == summoner_name:
            if player2 in summoners_dict:
                summoners_dict[player2] += 1
            else:
                summoners_dict[player2] = 0
        elif player2 == summoner_name:
            if player1 in summoners_dict:
                summoners_dict[player1] += 1
            else:
                summoners_dict[player1] = 0
        
    # Creating the graph
    network_graph = nx.Graph()
    network_graph.add_edges_from(players_list)
    pos = nx.layout.spring_layout(network_graph, k=0.5)

    # Creating the edges
    edge_trace = go.Scatter(
        x=[], y=[],
        line=dict(width=0.25, color='#888'),
        hoverinfo='none', mode='lines'
    )
    for edge in network_graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    # Creating the nodes
    node_trace = go.Scatter(
        x=[], y=[],
        text=[node for node in network_graph.nodes()],
        hoverinfo='text', mode='markers+text',
        marker=dict(
            colorscale='YlGnBu',
            color=[],
            size=25,
            line=dict(width=1)
        )
    )
    for node in network_graph.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    # Adding color and hovertext
    summoners_list = [player[0] for player in players_list] + [player[1] for player in players_list]
    for node in node_trace['text']:
        if node.lower() == summoner_name.lower():
            node_trace['marker']['color'] += tuple(['#3B81C8'])
        elif node in summoners_list:
            if node in summoners_dict and summoners_dict[node] >= 1:
                node_trace['marker']['color'] += tuple(['#FF8000'])
            else:
                node_trace['marker']['color'] += tuple(['#FFC080'])
        else:
            print('Error: ' + node + ' is not a summoner')
            raise
    
    # Creating the visualization
    network_fig = go.Figure(data=[edge_trace, node_trace],
                         layout=go.Layout(
                            title='<br>Players {}\'s played with'.format(summoner_name),
                            titlefont=dict(size=16),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=50,l=25,r=25,t=50),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)),
    )
    return network_fig

# Get summoner statistics
def get_summoner_stats(summoner_name):
    global TIER
    global RANK
    global role
    global similar_tab_content
    global older_tab_content
    
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

    champs_network_edges = []
    players_network_edges = []
    stats = []
    stats_older = []
    stats_recent = []
    i = 0
    teamId = None

    gameIds = []
    for match in matches['matches']:
        if get_role(match['role'], match['lane']) == role:
            gameIds.append((match['gameId'], match['champion']))
    
    for gameId, championId in gameIds[:N_MATCHES]:
        match_stats = {}
        match_stats['champion'] = champ_dict[str(championId)]
        match_stats_older = {}
        match_stats_recent = {}
        try:
            match_detail = watcher.match.by_id(REGION, gameId)
        except ApiError as err:
            if err.response.status_code == 429:
                return 429
            else:
                raise
        duration = match_detail['gameDuration'] / 60
        totalDamage = {match_detail['teams'][0]['teamId']: 0, match_detail['teams'][1]['teamId']: 0}
        totalDamage_older = {match_detail['teams'][0]['teamId']: 0, match_detail['teams'][1]['teamId']: 0}
        totalDamage_recent = {match_detail['teams'][0]['teamId']: 0, match_detail['teams'][1]['teamId']: 0}
        for participant in match_detail['participants']:
            if participant['championId'] == championId:
                teamId = participant['teamId']
                win = participant['stats']['win']
                K = participant['stats']['kills']
                D = participant['stats']['deaths']
                A = participant['stats']['assists']
                goldpm = participant['stats']['goldEarned'] / duration
                cspm = participant['stats']['totalMinionsKilled'] / duration
                damage = participant['stats']['totalDamageDealtToChampions']
                vision = participant['stats']['visionScore']
                
                match_stats['win'] = win
                match_stats['K'] = K
                match_stats['D'] = D
                match_stats['A'] = A
                match_stats['goldpm'] = goldpm
                match_stats['cspm'] = cspm
                match_stats['vision'] = vision
                
                if i < N_MATCHES // 2:
                    match_stats_older['win'] = win
                    match_stats_older['K'] = K
                    match_stats_older['D'] = D
                    match_stats_older['A'] = A
                    match_stats_older['goldpm'] = goldpm
                    match_stats_older['cspm'] = cspm
                    match_stats_older['damage'] = damage
                    match_stats_older['vision'] = vision
                else:
                    match_stats_recent['win'] = win
                    match_stats_recent['K'] = K
                    match_stats_recent['D'] = D
                    match_stats_recent['A'] = A
                    match_stats_recent['goldpm'] = goldpm
                    match_stats_recent['cspm'] = cspm
                    match_stats_recent['damage'] = damage
                    match_stats_recent['vision'] = vision
                
            totalDamage[participant['teamId']] += participant['stats']['totalDamageDealtToChampions']
            if i < N_MATCHES // 2:
                totalDamage_older[participant['teamId']] += participant['stats']['totalDamageDealtToChampions']
            else:
                totalDamage_recent[participant['teamId']] += participant['stats']['totalDamageDealtToChampions']
        match_stats['damage%'] = damage / totalDamage[participant['teamId']]
        stats.append(match_stats)
        champs_network_edges.append((summoner_name, match_stats['champion']))
        if i < N_MATCHES // 2:
            match_stats_older['damage%'] = match_stats_older['damage'] / totalDamage_older[participant['teamId']]
            stats_older.append(match_stats_older)
            i += 1
        else:
            match_stats_recent['damage%'] = match_stats_recent['damage'] / totalDamage_recent[participant['teamId']]
            stats_recent.append(match_stats_recent)
                
        for participant1 in match_detail['participantIdentities']:
            participant_sum_name1 = participant1['player']['summonerName']
            for participant2 in match_detail['participantIdentities']:
                participant_sum_name2 = participant2['player']['summonerName']
                if participant_sum_name1 != participant_sum_name2 and (participant_sum_name2, participant_sum_name1) not in players_network_edges:
                    players_network_edges.append((participant_sum_name1, participant_sum_name2))
    
    df_mean_player = pd.DataFrame(stats).mean()
    K, D, A = df_mean_player['K'], df_mean_player['D'], df_mean_player['A']
    player_win = df_mean_player['win'] * 100
    player_KDA = (K + A) / D if D > 0 else K + A
    player_vision = df_mean_player['vision']
    player_goldpm = df_mean_player['goldpm']
    player_cspm = df_mean_player['cspm']
    player_damage = df_mean_player['damage%'] * 100
    
    df_mean_player_older = pd.DataFrame(stats_older).mean()
    K_older, D_older, A_older = df_mean_player_older['K'], df_mean_player_older['D'], df_mean_player_older['A']
    player_win_older = df_mean_player_older['win'] * 100
    player_KDA_older = (K_older + A_older) / D_older if D_older > 0 else K_older + A_older
    player_vision_older = df_mean_player_older['vision']
    player_goldpm_older = df_mean_player_older['goldpm']
    player_cspm_older = df_mean_player_older['cspm']
    player_damage_older = df_mean_player_older['damage%'] * 100
    
    df_mean_player_recent = pd.DataFrame(stats_recent).mean()
    K_recent, D_recent, A_recent = df_mean_player_recent['K'], df_mean_player_recent['D'], df_mean_player_recent['A']
    player_win_recent = df_mean_player_recent['win'] * 100
    player_KDA_recent = (K_recent + A_recent) / D_recent if D_recent > 0 else K_recent + A_recent
    player_vision_recent = df_mean_player_recent['vision']
    player_goldpm_recent = df_mean_player_recent['goldpm']
    player_cspm_recent = df_mean_player_recent['cspm']
    player_damage_recent = df_mean_player_recent['damage%'] * 100
    
    # Get similar players stats
    csv = 'data/'
    csv += TIER.lower()[0]
    if RANK == 'I':
        csv += '1'
    elif RANK == 'II':
        csv += '2'
    elif RANK == 'III':
        csv += '3'
    elif RANK == 'IV':
        csv += '4'
    csv += '.csv'
    df_others = pd.read_csv(csv)
    df_others = df_others[df_others['role'] == role]
    for i, row in enumerate(df_others['champs']):
        for e in row.split(','):
            champ = e.strip('[').strip(']').strip().strip('\'')
            champs_network_edges.append(('Summoner{}'.format(i), champ))
    df_mean_others = df_others.mean()
    similar_win = df_mean_others['winrate']
    similar_KDA = df_mean_others['KDA']
    similar_vision = df_mean_others['vision']
    similar_goldpm = df_mean_others['goldpm']
    similar_cspm = df_mean_others['cspm']
    similar_damage = df_mean_others['damage%']
    
    stats_names = ['win rate', 'KDA', 'vision',
                   'gold per minute', 'creeps per minute', 'damage dealt ratio']

    max_stats = [100,
                 max(player_KDA, player_KDA_older, player_KDA_recent, similar_KDA),
                 max(player_vision, player_vision_older, player_vision_recent, similar_vision),
                 max(player_goldpm, player_goldpm_older, player_goldpm_recent, similar_goldpm),
                 max(player_cspm, player_cspm_older, player_cspm_recent, similar_cspm),
                 max(player_damage, player_damage_older, player_damage_recent, similar_damage)]

    player_stats = [player_win, player_KDA, player_vision,
                    player_goldpm, player_cspm, player_damage]
                    
    player_stats_older = [player_win_older, player_KDA_older, player_vision_older,
                          player_goldpm_older, player_cspm_older, player_damage_older]
                          
    player_stats_recent = [player_win_recent, player_KDA_recent, player_vision_recent,
                           player_goldpm_recent, player_cspm_recent, player_damage_recent]

    similar_stats = [similar_win, similar_KDA, similar_vision,
                     similar_goldpm, similar_cspm, similar_damage]

    # Similar players comparison tab
    df_radar_chart_similar = pd.DataFrame(dict(r=[player_stats[i] / max_stats[i] for i in range(len(max_stats))] +
                        [similar_stats[i] / max_stats[i] for i in range(len(max_stats))],
                     theta=stats_names * 2,
                     player=[summoner_name] * 6 + ['similar players'] * 6))
    radar_chart_similar = px.line_polar(data_frame=df_radar_chart_similar, r='r', theta='theta', color='player',
                               color_discrete_sequence=COLOR_SEQ,
                               line_close=True, title=summoner_name + " vs similar players statistics")
    radar_chart_similar.update_traces(fill='toself')
    
    df = pd.DataFrame({'player': [summoner_name, 'similar players'],
                       'win rate': [player_win, similar_win],
                       'KDA': [player_KDA, similar_KDA],
                       'vision': [player_vision, similar_vision],
                       'gold per minute': [player_goldpm, similar_goldpm],
                       'creeps per minute': [player_cspm, similar_cspm],
                       'damage dealt ratio': [player_damage, similar_damage]})
    
    win_rate = px.bar(df, x='win rate', y='player', orientation='h', labels={'player':''},
                       color='player', color_discrete_sequence=COLOR_SEQ)
    win_rate.update_layout(yaxis=dict(showticklabels=False), showlegend=False,
                           autosize=False, width=300, height=200, margin=dict(l=0, r=0, t=0, b=50, pad=0))
    
    KDA = px.bar(df, x='KDA', y='player', orientation='h', labels={'player':''},
                       color='player', color_discrete_sequence=COLOR_SEQ)
    KDA.update_layout(yaxis=dict(showticklabels=False), showlegend=False,
                      autosize=False, width=300, height=200, margin=dict(l=0, r=0, t=0, b=50, pad=0))
    
    vision = px.bar(df, x='vision', y='player', orientation='h', labels={'player':''},
                       color='player', color_discrete_sequence=COLOR_SEQ)
    vision.update_layout(yaxis=dict(showticklabels=False), showlegend=False,
                         autosize=False, width=300, height=200, margin=dict(l=0, r=0, t=0, b=50, pad=0))
    
    goldpm = px.bar(df, x='gold per minute', y='player', orientation='h', labels={'player':''},
                       color='player', color_discrete_sequence=COLOR_SEQ)
    goldpm.update_layout(yaxis=dict(showticklabels=False), showlegend=False,
                         autosize=False, width=300, height=200, margin=dict(l=0, r=0, t=0, b=50, pad=0))
    
    cspm = px.bar(df, x='creeps per minute', y='player', orientation='h', labels={'player':''},
                       color='player', color_discrete_sequence=COLOR_SEQ)
    cspm.update_layout(yaxis=dict(showticklabels=False), showlegend=False,
                       autosize=False, width=300, height=200, margin=dict(l=0, r=0, t=0, b=50, pad=0))
    
    damage = px.bar(df, x='damage dealt ratio', y='player', orientation='h', labels={'player':''},
                       color='player', color_discrete_sequence=COLOR_SEQ)
    damage.update_layout(yaxis=dict(showticklabels=False), showlegend=False,
                         autosize=False, width=300, height=200, margin=dict(l=0, r=0, t=0, b=50, pad=0))
    
    champs_network_graph = create_champs_network_graph(champs_network_edges)
    
    # Older matches comparison tab
    df_radar_chart_older = pd.DataFrame(dict(r=[player_stats_recent[i] / max_stats[i] for i in range(len(max_stats))] +
                        [player_stats_older[i] / max_stats[i] for i in range(len(max_stats))],
                        theta=stats_names * 2,
                        matches=['{} most recent matches'.format(N_MATCHES//2)] * 6 + ['{} previous matches'.format(N_MATCHES//2)] * 6))
    radar_chart_older = px.line_polar(data_frame=df_radar_chart_older, r='r', theta='theta', color='matches',
                                color_discrete_sequence=COLOR_SEQ,
                                line_close=True, title="Older vs most recent matches statistics")
    radar_chart_older.update_traces(fill='toself')
    
    df_older = pd.DataFrame({'matches': ['{} most recent matches'.format(N_MATCHES//2), '{} previous matches'.format(N_MATCHES//2)],
                       'win rate': [player_win_recent, player_win_older],
                       'KDA': [player_KDA_recent, player_KDA_older],
                       'vision': [player_vision_recent, player_vision_older],
                       'gold per minute': [player_goldpm_recent, player_goldpm_older],
                       'creeps per minute': [player_cspm_recent, player_cspm_older],
                       'damage dealt ratio': [player_damage_recent, player_damage_older]})
    
    win_rate_older = px.bar(df_older, x='win rate', y='matches', orientation='h', labels={'matches':''},
                       color='matches', color_discrete_sequence=COLOR_SEQ)
    win_rate_older.update_layout(yaxis=dict(showticklabels=False), showlegend=False,
                           autosize=False, width=300, height=200, margin=dict(l=0, r=0, t=0, b=50, pad=0))
    
    KDA_older = px.bar(df_older, x='KDA', y='matches', orientation='h', labels={'matches':''},
                       color='matches', color_discrete_sequence=COLOR_SEQ)
    KDA_older.update_layout(yaxis=dict(showticklabels=False), showlegend=False,
                      autosize=False, width=300, height=200, margin=dict(l=0, r=0, t=0, b=50, pad=0))
    
    vision_older = px.bar(df_older, x='vision', y='matches', orientation='h', labels={'matches':''},
                       color='matches', color_discrete_sequence=COLOR_SEQ)
    vision_older.update_layout(yaxis=dict(showticklabels=False), showlegend=False,
                         autosize=False, width=300, height=200, margin=dict(l=0, r=0, t=0, b=50, pad=0))
    
    goldpm_older = px.bar(df_older, x='gold per minute', y='matches', orientation='h', labels={'matches':''},
                       color='matches', color_discrete_sequence=COLOR_SEQ)
    goldpm_older.update_layout(yaxis=dict(showticklabels=False), showlegend=False,
                         autosize=False, width=300, height=200, margin=dict(l=0, r=0, t=0, b=50, pad=0))
    
    cspm_older = px.bar(df_older, x='creeps per minute', y='matches', orientation='h', labels={'matches':''},
                       color='matches', color_discrete_sequence=COLOR_SEQ)
    cspm_older.update_layout(yaxis=dict(showticklabels=False), showlegend=False,
                       autosize=False, width=300, height=200, margin=dict(l=0, r=0, t=0, b=50, pad=0))
    
    damage_older = px.bar(df_older, x='damage dealt ratio', y='matches', orientation='h', labels={'matches':''},
                       color='matches', color_discrete_sequence=COLOR_SEQ)
    damage_older.update_layout(yaxis=dict(showticklabels=False), showlegend=False,
                         autosize=False, width=300, height=200, margin=dict(l=0, r=0, t=0, b=50, pad=0))
    
    players_network_graph = create_players_network_graph(summoner_name, players_network_edges)
    
    # Create tabs content
    similar_tab_content = [
            dcc.Graph(
                id='player-vs-similar-stats',
                figure=radar_chart_similar
            ),
            html.Div([
                dcc.Graph(id='win-rate', figure=win_rate),
                dcc.Graph(id='goldpm', figure=goldpm),
                dcc.Graph(id='KDA', figure=KDA),
                dcc.Graph(id='cspm', figure=cspm),
                dcc.Graph(id='vision', figure=vision),
                dcc.Graph(id='damage', figure=damage)
            ],
            style={'columnCount': 3}
            ),
            html.Div(
                dcc.Graph(id='network', figure=champs_network_graph)
            )]
            
    older_tab_content = [
            dcc.Graph(
                id='player-vs-similar-stats',
                figure=radar_chart_older
            ),
            html.Div([
                dcc.Graph(id='win-rate', figure=win_rate_older),
                dcc.Graph(id='goldpm', figure=goldpm_older),
                dcc.Graph(id='KDA', figure=KDA_older),
                dcc.Graph(id='cspm', figure=cspm_older),
                dcc.Graph(id='vision', figure=vision_older),
                dcc.Graph(id='damage', figure=damage_older)
            ],
            style={'columnCount': 3}
            ),
            html.Div(
                dcc.Graph(id='network', figure=players_network_graph)
            )]
    return (similar_tab_content, older_tab_content)



# Dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config['suppress_callback_exceptions'] = True

markdown_text = '''
League of Legends Dashboard tool developed by Laura Almón Manzano for a University project.
Done with a Riot's personal API, which only supports 20 requests per second or 100 every 2 minutes.
To use it, enter a summoner's name and wait for the graphs to update. Please, be patient!
You can use mine: "Láudano". Hope you like it!
'''

app.layout = html.Div([
    html.H1('League of Legends Dashboard', style={'textAlign': 'center'}
    ),
    dcc.Markdown(markdown_text
    ),
    html.Div([
        html.Div('Summoner\'s name:'
        ),
        dcc.Input(id='summoner-name-input', type='search', placeholder='Summoner\'s name', debounce=True
        ),
        html.Div(id='summoner-name-error-output'
        )]
    ),
    dcc.Loading(id='loading', type='default', children=html.Div(id='loading-output')
    ),
    html.Div([
        html.H2(id='summoner-name-output', style={'textAlign': 'center'}
        ),
        html.Div([
            html.Div(id='tier'), html.Div(id='rank'), html.Div(id='role')
            ],
            style={'columnCount': 3, 'textAlign': 'center'}
        ),
        dcc.Tabs(id='tabs', value='similar_tab', children=[
            dcc.Tab(label='Similar players comparison', value='similar_tab'),
            dcc.Tab(label='Older matches comparison', value='older_tab')
        ]),
        html.Div(id='tabs-content', children=''
        )]
    )]
)

# Update tabs content
@app.callback(
    [Output('loading-output', 'children'),
    Output('summoner-name-output', 'children'),
    Output('summoner-name-error-output', 'children'),
    Output('tier', 'children'),
    Output('rank', 'children'),
    Output('role', 'children'),
    Output('tabs-content', 'children')],
    [Input('summoner-name-input', 'value'),
    Input('tabs', 'value')]
)
def update_summoner_name(summoner_name, tab):
    global current_tab
    
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == '':
        return ['', '', '', '', '', '', '']
    
    elif button_id == 'tabs':
        if tab == 'similar_tab':
            current_tab = 'similar'
        elif tab == 'older_tab':
            current_tab = 'older'
        if summoner_name in ['', None]:
            return ['', '', '', '', '', '', '']
        elif tab == 'similar_tab':
            return ['', summoner_name + ' statistics', '',
                [html.P('TIER'), html.H6(TIER)], [html.P('RANK'), html.H6(RANK)], [html.P('ROLE'), html.H6(role)],
                similar_tab_content]
        elif tab == 'older_tab':
            return ['', summoner_name + ' statistics', '',
                [html.P('TIER'), html.H6(TIER)], [html.P('RANK'), html.H6(RANK)], [html.P('ROLE'), html.H6(role)],
                older_tab_content]

    elif button_id == 'summoner-name-input':
        if summoner_name in ['', None]:
            return ['', '', '', '', '', '', '']
        summoner_stats = get_summoner_stats(summoner_name)
        if summoner_stats is None:
            return ['', '', 'Summoner name not found.', '', '', '', '']
        elif summoner_stats == 429:
            return ['', '', 'Too many requests. Try again in a few seconds.', '', '', '', '']
        
        if None in [similar_tab_content, older_tab_content]:
            return ['', '', 'Error. Try again in a few seconds.', '', '', '', '']
        
        if current_tab == 'similar':
            return ['', summoner_name + ' statistics', '',
                [html.P('TIER'), html.H6(TIER)], [html.P('RANK'), html.H6(RANK)], [html.P('ROLE'), html.H6(role)],
                similar_tab_content]
        elif current_tab == 'older':
            return ['', summoner_name + ' statistics', '',
                [html.P('TIER'), html.H6(TIER)], [html.P('RANK'), html.H6(RANK)], [html.P('ROLE'), html.H6(role)],
                older_tab_content]

if __name__ == '__main__':
    app.run_server(debug=True)