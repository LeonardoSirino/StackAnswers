import re
from typing import List

import requests

YOUR_BOT_ID = 'your_bot_id'
YOUR_BOT_KEY = 'your_bot_key'

TELEGRAM_API_URL = f'https://api.telegram.org/{YOUR_BOT_ID}:{YOUR_BOT_KEY}'
CHAT_ID = -1


def get_ids() -> List[str]:
    res = requests.get(url=f'{TELEGRAM_API_URL}/getUpdates')

    content = res.json()['result']

    messages = filter(lambda x: x['message']['chat']['id'] == CHAT_ID, content)
    messages_content = [x['message']['text'] for x in messages]

    regex = r'Apples id: (\d+)'

    apples_id = []
    for x in messages_content:
        res = re.search(regex, x)
        if res:
            apples_id.append(res.groups()[0])

    return apples_id


if __name__ == '__main__':
    ids = get_ids()

    print(ids)
