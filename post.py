
neurons = range(1,15)


for neuron in neurons:

    run_command=f"pm2 start --name sn5-{neuron} \"python -m miner.main --port 5001 --wallet-name sn5 --wallet-hotkey h{neuron}\""

    port = 5000 + neuron
    reg_id_command=f"python tools/post_ip_chain.py --wallet-name sn5 --hotkey h{neuron} --ip 116.202.117.116 --port {port}"
    print("-----")
    print(run_command)
    print(reg_id_command)
    print("-----")
  