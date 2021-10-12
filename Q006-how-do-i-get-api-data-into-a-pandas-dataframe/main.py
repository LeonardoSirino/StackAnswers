import json

import pandas as pd

from dataclasses import dataclass


@dataclass
class BettingData:
    away_team: str
    home_team: str
    spread: str
    overUnder: str


json_data = json.loads(open(
    'Q006-how-do-i-get-api-data-into-a-pandas-dataframe\sample_data.json', 'r').read())

content = []
for entry in json_data:
    for line in entry['lines']:
        data = BettingData(away_team=entry['away_team'],
                           home_team=entry['home_team'],
                           spread=line['spread'],
                           overUnder=line['overUnder'])

        content.append(data)


df = pd.DataFrame(content)

print(df)
