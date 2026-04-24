#!/usr/bin/env bash
# Hone container entrypoint.
#
# Reads NODE_TYPE + a handful of HONE_* env vars and assembles the right
# python command. Two modes:
#
#   NODE_TYPE=validator
#       runs ``python neurons/validator.py`` with --wallet/--netuid/...
#
#   NODE_TYPE=miner
#       runs ``python neurons/miner.py`` with PP-stage args. The PP
#       transport peer is resolved by Docker DNS (the container's
#       compose ``hostname:`` field) so each miner-pair stays on its
#       own service-name pair (e.g. miner1-s0 <-> miner1-s1) and
#       ports can be the same across miners (each container is in its
#       own netns).
#
# Anything after ``--`` on the docker command line is appended verbatim
# so per-environment overrides ('--debug', '--test', ...) are easy.

set -euo pipefail

NODE_TYPE="${NODE_TYPE:?NODE_TYPE must be set to 'validator' or 'miner'}"
NETUID="${NETUID:-268}"
WALLET_NAME="${WALLET_NAME:?WALLET_NAME must be set}"
WALLET_HOTKEY="${WALLET_HOTKEY:?WALLET_HOTKEY must be set}"
SUBTENSOR_NETWORK="${SUBTENSOR_NETWORK:-finney}"
SUBTENSOR_CHAIN_ENDPOINT="${SUBTENSOR_CHAIN_ENDPOINT:-}"
EXTRA_FLAGS="${HONE_EXTRA_FLAGS:-}"

# bittensor's wallet config args expect ``--wallet.name`` / ``--wallet.hotkey``.
# We always pass them; the user is responsible for mounting
# ~/.bittensor/wallets into the container (see compose volumes:).
COMMON_FLAGS=(
    --netuid "${NETUID}"
    --wallet.name "${WALLET_NAME}"
    --wallet.hotkey "${WALLET_HOTKEY}"
    --subtensor.network "${SUBTENSOR_NETWORK}"
)

# Optional explicit chain endpoint (e.g. for a local subtensor or
# alternative finney mirror). Only added when set so the bittensor
# client falls back to the network-default endpoint otherwise.
if [[ -n "${SUBTENSOR_CHAIN_ENDPOINT}" ]]; then
    COMMON_FLAGS+=(--subtensor.chain_endpoint "${SUBTENSOR_CHAIN_ENDPOINT}")
fi

case "${NODE_TYPE}" in
    validator)
        # Validator currently always runs as a single process (FSDP
        # within the process via dp_shard from hparams.json). We don't
        # use torchrun: the validator pins ``pp_degree=1`` and the
        # in-process FSDP setup handles multi-GPU sharding via
        # CUDA_VISIBLE_DEVICES picking the right devices for this
        # container.
        echo "[entrypoint] starting validator: netuid=${NETUID} wallet=${WALLET_NAME}/${WALLET_HOTKEY}"
        exec python /app/neurons/validator.py \
            "${COMMON_FLAGS[@]}" \
            ${EXTRA_FLAGS}
        ;;

    miner)
        PP_STAGE="${PP_STAGE:-0}"
        PP_NUM_STAGES="${PP_NUM_STAGES:-1}"
        PP_PEER_HOST_PREV="${PP_PEER_HOST_PREV:-127.0.0.1}"
        PP_PEER_HOST_NEXT="${PP_PEER_HOST_NEXT:-127.0.0.1}"
        PP_PEER_PORT_BASE_PREV="${PP_PEER_PORT_BASE_PREV:-50100}"
        PP_PEER_PORT_BASE_NEXT="${PP_PEER_PORT_BASE_NEXT:-50100}"
        PP_LISTEN_HOST="${PP_LISTEN_HOST:-0.0.0.0}"
        PP_LISTEN_PORT_BASE="${PP_LISTEN_PORT_BASE:-50000}"

        MINER_FLAGS=(
            "${COMMON_FLAGS[@]}"
            --pp-stage "${PP_STAGE}"
            --pp-num-stages "${PP_NUM_STAGES}"
            --pp-peer-host-prev "${PP_PEER_HOST_PREV}"
            --pp-peer-host-next "${PP_PEER_HOST_NEXT}"
            --pp-peer-port-base-prev "${PP_PEER_PORT_BASE_PREV}"
            --pp-peer-port-base-next "${PP_PEER_PORT_BASE_NEXT}"
            --pp-listen-host "${PP_LISTEN_HOST}"
            --pp-listen-port-base "${PP_LISTEN_PORT_BASE}"
        )

        # Single-rank-per-stage path (NPROC_PER_NODE=1, no torchrun).
        # When the user wants intra-stage FSDP sharding they bump
        # NPROC_PER_NODE >= 2 AND map that many GPUs into the
        # container; we fall through to torchrun in that branch so the
        # NCCL world is built correctly.
        NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
        if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
            echo "[entrypoint] starting miner via torchrun: pp_stage=${PP_STAGE}/${PP_NUM_STAGES} nproc=${NPROC_PER_NODE} wallet=${WALLET_NAME}/${WALLET_HOTKEY}"
            exec torchrun \
                --nproc_per_node="${NPROC_PER_NODE}" \
                --master_addr=127.0.0.1 \
                --master_port="${MASTER_PORT:-29500}" \
                /app/neurons/miner.py \
                "${MINER_FLAGS[@]}" \
                ${EXTRA_FLAGS}
        else
            echo "[entrypoint] starting miner: pp_stage=${PP_STAGE}/${PP_NUM_STAGES} wallet=${WALLET_NAME}/${WALLET_HOTKEY} listen=${PP_LISTEN_HOST}:${PP_LISTEN_PORT_BASE} next=${PP_PEER_HOST_NEXT}:${PP_PEER_PORT_BASE_NEXT}"
            exec python /app/neurons/miner.py \
                "${MINER_FLAGS[@]}" \
                ${EXTRA_FLAGS}
        fi
        ;;

    *)
        echo "[entrypoint] unknown NODE_TYPE='${NODE_TYPE}', expected 'validator' or 'miner'" >&2
        exit 64
        ;;
esac
