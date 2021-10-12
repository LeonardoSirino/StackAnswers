import json

data = """
{
  "data": [
    {
      "id": "xxxx",
      "account_type": "None",
      "description": "Lorem Ipsum",
      "engagement_score": "xx",
      "jumpers": "xxxxx",
      "friends": "xxx",
      "global": "xxxxxxx",
      "hidden": true,
      "location": "xxxx, xx",
      "name": "your_name"
    }
  ]
}

"""

json_data = json.loads(data)

print(json_data['data'][0]['name'])


data = [
    [1, 2, 3],
    [4, 5, 6]
]

print(data[0][0])
