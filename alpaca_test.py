import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
load_dotenv()
client = TradingClient(api_key=os.getenv("APCA_API_KEY_ID"), secret_key=os.getenv("APCA_API_SECRET_KEY"), paper=True)
account = client.get_account()
print(f"Connected successfully! Paper Portfolio Cash: ${account.cash}")
