import requests

def get_climber_id(climber):
    cookies = {
        '_verticallife_resultservice_session': '6Si8J07mNZHWub3y%2Bn8Mve2Bh67NlJ0BVJA%2F5ICI4eH5PC93EoqjMT841bizyl0bbokYBnB9SUOsOEHSebSZJcBQ7xiTrc6XwKqUHvBYiLSFP84Vpf7cAqsdrmyLQBspqZbvZY4DWZc7NZ2R1vKEzmKfT%2Bw2Zg%2BG7CyPa7QjfR2gzxyCOXvP8xjI9%2FsOivPmTKLLphwPn5nY5OGNrixlEjFqUfdqjAWCQ54Ko4u4fup3Og5EV%2FqWlBZVUtFPCK3px%2Fqb26%2B8NuZqF6i5Xm4B0GVB3%2B0Mbzs4UyiBH%2BpLqxeoMDYf%2BoCRY9fsIA%3D%3D--qAkB%2FXsehQCDuaXf--ePqQOMGsYwt2LWEImdKKiA%3D%3D',
    }

    headers = {
        'accept': 'application/json',
        'accept-language': 'en-US,en;q=0.9',
        'cache-control': 'no-cache',
        'dnt': '1',
        'pragma': 'no-cache',
        'priority': 'u=1, i',
        'referer': 'https://ifsc.results.info/',
        'sec-ch-ua': '"Not;A=Brand";v="24", "Chromium";v="128"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
        'x-csrf-token': 'hixPoCsgyJ045LS7ykw2Gw4KvN_xrjynRF4egoQkWuuWKT5pM_jX7SoTtsKLzCyiA5fhjRfEzQ3bQ6mL9cr_YQ',
    }

    params = {
        'name': climber,
    }

    response = requests.get('https://ifsc.results.info/api/v1/athletes', params=params, cookies=cookies, headers=headers)
    return response.json()[0]['id']

def get_climber_height(id):
    cookies = {
        '_verticallife_resultservice_session': 'OMMdlDQEAqwTffzNo%2BPASxArSGcCdAzSC%2FmjMeSa8EDkGKPF%2FyrJd5qrC3UndiD3SAhpHI%2F9qanVJHXGMyjDi6yjRW5ofVq6sbu728IfwnV3wY4NqlVps2Jop92LsUmDye8xWn9SazVYtUq7hkZ2Kdi0YY1uqINwzjI6YlgS8W66ffmqNgGU9C9WtMgMyEM0XZbIPeEf4qXSNODP6DQRzwdo%2BWlGjgzp38L7BfmIKlhhBffFGt6yclBqfKtlT0PiuP2ryeqswf9N5CVrcN0Q5RgQXDs%2BIdyGRatdkyxsuazM8M%2BC9koA3HFU8w%3D%3D--e7HK8BaG4gglaQsZ--%2BPPT2ThVznLcAXlHNcKFgw%3D%3D',
    }

    headers = {
        'accept': 'application/json',
        'accept-language': 'en-US,en;q=0.9',
        'cache-control': 'no-cache',
        'dnt': '1',
        'pragma': 'no-cache',
        'priority': 'u=1, i',
        'referer': 'https://ifsc.results.info/',
        'sec-ch-ua': '"Not;A=Brand";v="24", "Chromium";v="128"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
        'x-csrf-token': 'Bn08ImBGGGF_nTG6PY1rZYCx1OiDa9VbXNaPWcogQqoWeE3reJ4HEW1qM8N8DXHcjSyJumUBJPHDyzhQu87nIA',
    }

    response = requests.get(f'https://ifsc.results.info/api/v1/athletes/{id}', cookies=cookies, headers=headers)
    return response.json()['height']

if __name__ == '__main__':
    import pandas as pd
    import torch
    from pmf import PMF
    from lr import LogReg

    pmf_model = torch.load(f"models/pmf/model_rl_{100}_d_{3}_full_data.pth")
    pmf_model.eval()
    Climbers_rl_100 = pmf_model.climber_vocab.get_itos()[1:]

    climber_ids = {climber:get_climber_id(climber) for climber in Climbers_rl_100}
    climber_heights = {climber:get_climber_height(ID) for climber, ID in climber_ids.items()}

    climber_data = pd.DataFrame({
        "Name": climber_ids.keys(),
        "ID": climber_ids.values(),
        "Height": climber_heights.values(),
    })
    climber_data.dropna(inplace=True)
    climber_data.to_csv('data/climbers_heights.csv')

