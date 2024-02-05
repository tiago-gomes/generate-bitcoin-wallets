import os
import hashlib
import base58
import requests
import sqlite3
from multiprocessing import Process

def generate_wallet():
    private_key = os.urandom(32)
    extended_private_key = hashlib.sha256(private_key).digest()
    address = base58.b58encode_check(b"\x00" + hashlib.new("ripemd160", hashlib.sha256(b"\x04" + hashlib.sha256(extended_private_key).digest()).digest()).digest()).decode()
    return address, private_key.hex()

def check_balance(bitcoin_address):
    api_url = f"https://api.blockchain.info/haskoin-store/btc/address/{bitcoin_address}/balance"
    try:
        response = requests.get(api_url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data['confirmed'] / 100000000
    except requests.exceptions.RequestException as e:
        print(f"Error checking balance for {bitcoin_address}: {e}")
    return None

def save_wallet_to_database(address, private_key, balance):
    conn = sqlite3.connect('wallets.db')
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS wallets
                      (address TEXT, private_key TEXT, balance REAL)''')

    # Insert the wallet into the table
    cursor.execute("INSERT INTO wallets VALUES (?, ?, ?)", (address, private_key, balance))
    conn.commit()

    print("Wallet saved to the database.")

    conn.close()

def generate_and_save_wallets():
    while True:
        wallet = generate_wallet()
        address, _ = wallet
        balance = check_balance(address)
        if balance > 0:
            print(f"Balance: {balance} BTC" if balance is not None else "Failed to retrieve balance information.")
            save_wallet_to_database(address, *wallet)

if __name__ == '__main__':
    num_processes = 5  # Set the desired number of processes

    processes = []
    for _ in range(num_processes):
        process = Process(target=generate_and_save_wallets)
        process.start()
        processes.append(process)