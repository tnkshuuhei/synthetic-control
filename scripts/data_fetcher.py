import requests
import json
import os
from pathlib import Path

def fetch_data():
    """GrowthePieからデータを取得"""
    url = 'https://api.growthepie.xyz/v1/fundamentals_full.json'
    response = requests.get(url)
    
    # データディレクトリの確認と作成
    data_dir = Path('data/_local')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # データの保存
    data_file = data_dir / 'fundamentals_full.json'
    with open(data_file, 'w') as f:
        json.dump(response.json(), f)
        
    print(f"Data saved to {data_file}")

if __name__ == "__main__":
    fetch_data()