import argparse
import json
import sys
import socket
from pathlib import Path
from substrateinterface import SubstrateInterface, Keypair

import netaddr


def load_keypair(wallet_name: str, hotkey: str, wallet_path: str = "~/.bittensor/wallets") -> Keypair:
    """Load keypair from Bittensor wallet files."""
    base_path = Path(wallet_path).expanduser()
    hotkey_file = base_path / wallet_name / "hotkeys" / hotkey
    
    if not hotkey_file.exists():
        raise FileNotFoundError(f"Hotkey file not found: {hotkey_file}")
    
    with open(hotkey_file, "r") as f:
        keypair_data = json.load(f)
    
    if "secretSeed" in keypair_data:
        return Keypair.create_from_seed(keypair_data["secretSeed"])
    elif "secretKey" in keypair_data:
        return Keypair.create_from_seed(keypair_data["secretKey"])
    else:
        raise ValueError("Could not find secret key in hotkey file")


def load_coldkey_address(wallet_name: str, wallet_path: str = "~/.bittensor/wallets") -> str:
    """Load coldkey address from wallet."""
    base_path = Path(wallet_path).expanduser()
    coldkey_file = base_path / wallet_name / "coldkeypub.txt"
    
    if coldkey_file.exists():
        with open(coldkey_file, "r") as f:
            return f.read().strip()
    
    coldkey_file = base_path / wallet_name / "coldkey"
    if coldkey_file.exists():
        with open(coldkey_file, "r") as f:
            data = json.load(f)
            return data.get("ss58Address", "")
    
    print(f"Warning: Could not find coldkey for wallet {wallet_name}, using hotkey address")
    return ""


def resolve_hostname_to_ip(hostname: str) -> str:
    """Resolve a hostname to an IP address."""
    if netaddr:
        try:
            netaddr.IPAddress(hostname)
            return hostname
        except netaddr.AddrFormatError:
            pass
    else:
        try:
            parts = hostname.split('.')
            if len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
                return hostname
        except:
            pass
    
    try:
        ip_address = socket.gethostbyname(hostname)
        print(f"Resolved '{hostname}' to '{ip_address}'")
        return ip_address
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve hostname '{hostname}': {e}")


def ip_to_int(ip_str: str) -> int:
    """Convert IP address to integer."""
    if netaddr:
        return int(netaddr.IPAddress(ip_str))
    else:
        parts = ip_str.split('.')
        if len(parts) != 4:
            raise ValueError("Invalid IP format")
        
        for part in parts:
            if not 0 <= int(part) <= 255:
                raise ValueError("IP octet out of range")
        
        return int(''.join([f"{int(x):02x}" for x in parts]), 16)


def ip_version(ip_str: str) -> int:
    """Get IP version (4 for IPv4, 6 for IPv6)."""
    if netaddr:
        return int(netaddr.IPAddress(ip_str).version)
    else:
        if ':' in ip_str:
            return 6
        else:
            return 4


def set_miner_ip(
    wallet_name: str,
    hotkey: str,
    ip: str,
    port: int,
    netuid: int = 1,
    chain_endpoint: str = "wss://entrypoint-finney.opentensor.ai:443",
    wallet_path: str = "~/.bittensor/wallets",
    include_coldkey: bool = True,
    protocol: int = 4
) -> bool:
    """Set miner IP address on chain."""
    
    print(f"Loading keypair for {wallet_name}/{hotkey}...")
    try:
        keypair = load_keypair(wallet_name, hotkey, wallet_path)
        print(f"✓ Loaded keypair with address: {keypair.ss58_address}")
    except Exception as e:
        print(f"✗ Failed to load keypair: {e}")
        return False
    
    coldkey_address = ""
    if include_coldkey:
        coldkey_address = load_coldkey_address(wallet_name, wallet_path)
        if not coldkey_address:
            coldkey_address = keypair.ss58_address
    
    print(f"Resolving IP {ip}...")
    try:
        resolved_ip = resolve_hostname_to_ip(ip)
        ip_int = ip_to_int(resolved_ip)
        ip_ver = ip_version(resolved_ip)
        print(f"✓ IP resolved: {ip} -> {resolved_ip} (IPv{ip_ver}, int: {ip_int})")
    except Exception as e:
        print(f"✗ Failed to process IP: {e}")
        return False
    
    print(f"Connecting to chain at {chain_endpoint}...")
    try:
        substrate = SubstrateInterface(url=chain_endpoint)
        print(f"✓ Connected to chain")
    except Exception as e:
        print(f"✗ Failed to connect to chain: {e}")
        return False
    
    print(f"Creating serve_axon transaction...")
    try:
        params = {
            'netuid': netuid,
            'version': 1,
            'ip': ip_int,
            'port': port,
            'ip_type': ip_ver,
            'protocol': protocol,
        }
        
        if include_coldkey:
            params.update({
                'hotkey': keypair.ss58_address,
                'coldkey': coldkey_address,
                'placeholder1': 0,
                'placeholder2': 0,
            })
        
        print(f"  Parameters: {params}")
        
        call = substrate.compose_call(
            call_module='SubtensorModule',
            call_function='serve_axon',
            call_params=params
        )
        
        extrinsic = substrate.create_signed_extrinsic(call=call, keypair=keypair)
        print(f"✓ Transaction created")
    except Exception as e:
        print(f"✗ Failed to create transaction: {e}")
        substrate.close()
        return False
    
    print(f"Submitting transaction to chain...")
    try:
        receipt = substrate.submit_extrinsic(
            extrinsic, 
            wait_for_inclusion=True, 
            wait_for_finalization=True
        )
        
        if hasattr(receipt, 'process_events'):
            receipt.process_events()
        
        if receipt.is_success:
            print(f"✓ Successfully set miner IP!")
            print(f"  - Miner hotkey: {keypair.ss58_address}")
            print(f"  - IP address: {resolved_ip}:{port}")
            print(f"  - Netuid: {netuid}")
            print(f"  - Block hash: {receipt.block_hash}")
            return True
        else:
            print(f"✗ Transaction failed!")
            if hasattr(receipt, 'error_message'):
                print(f"  Error: {receipt.error_message}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to submit transaction: {e}")
        return False
    finally:
        substrate.close()


def main():
    parser = argparse.ArgumentParser(
        description="Set miner IP address on Bittensor chain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            python set_miner_ip.py --wallet-name default --hotkey miner --ip 1.2.3.4 --port 8091
            python set_miner_ip.py --wallet-name my_wallet --hotkey my_miner --ip example.com --port 8092 --netuid 42
        """
    )
    
    parser.add_argument(
        "--wallet-name",
        type=str,
        required=True,
        help="Name of the wallet (e.g., 'default')"
    )
    
    parser.add_argument(
        "--hotkey",
        type=str,
        required=True,
        help="Name of the hotkey (e.g., 'miner')"
    )
    
    parser.add_argument(
        "--ip",
        type=str,
        required=True,
        help="Public IP address or hostname of the miner (e.g., '1.2.3.4' or 'example.com')"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port number for the miner API (e.g., 8091)"
    )
    
    parser.add_argument(
        "--netuid",
        type=int,
        default=5,
        help="Network UID (default: 5)"
    )
    
    parser.add_argument(
        "--chain-endpoint",
        type=str,
        default="wss://entrypoint-finney.opentensor.ai:443",
        help="Chain endpoint URL (default: finney)"
    )
    
    parser.add_argument(
        "--wallet-path",
        type=str,
        default="~/.bittensor/wallets",
        help="Path to wallets directory (default: ~/.bittensor/wallets)"
    )
    
    parser.add_argument(
        "--protocol",
        type=int,
        default=4,
        help="Protocol type (default: 4)"
    )
    
    parser.add_argument(
        "--no-coldkey",
        action="store_true",
        help="Don't include coldkey parameters (for older chain versions)"
    )
    
    args = parser.parse_args()
    
    success = set_miner_ip(
        wallet_name=args.wallet_name,
        hotkey=args.hotkey,
        ip=args.ip,
        port=args.port,
        netuid=args.netuid,
        chain_endpoint=args.chain_endpoint,
        wallet_path=args.wallet_path,
        include_coldkey=not args.no_coldkey,
        protocol=args.protocol
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()