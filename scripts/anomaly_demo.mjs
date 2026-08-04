#!/usr/bin/env node
/**
 * AnomalyDetection lifecycle demo (studionet).
 *
 * Deploys the contract (or uses ANOMALY_CONTRACT_ADDRESS if set), then drives
 * the full lifecycle: novelty detection, near-duplicate classification,
 * nearest-match retrieval, and removal.
 *
 * Run:   node scripts/anomaly_demo.mjs
 * Reuse a deployed contract:   ANOMALY_CONTRACT_ADDRESS=0x... node scripts/anomaly_demo.mjs
 *
 * Notes:
 * - Signs with a freshly generated account (no keystore needed).
 * - Assertions are semantic (same text => duplicate, clearly-different text =>
 *   novel), not exact-similarity based, since embeddings run in the real VM.
 * - Polls conservatively to respect Studio's free-tier RPC rate limit.
 */
import { createClient, createAccount, generatePrivateKey } from "genlayer-js";
import { readFileSync } from "fs";
import { studionet } from "genlayer-js/chains";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const CONTRACT_PATH = new URL("../contracts/anomaly_detection.py", import.meta.url);
const WAIT_RETRIES = 120;
const WAIT_INTERVAL = 5000;

function step(label) {
  console.log(`\n--- ${label}`);
}

async function main() {
  const pk = generatePrivateKey();
  const acc = createAccount(pk);
  console.log(`signer: ${acc.address}`);

  const client = createClient({ chain: studionet, account: acc });

  let address = process.env.ANOMALY_CONTRACT_ADDRESS;
  if (address) {
    console.log(`using existing contract: ${address}`);
  } else {
    step("deploy");
    const code = readFileSync(CONTRACT_PATH);
    const hash = await client.deployContract({ code, args: [], account: acc });
    const receipt = await client.waitForTransactionReceipt({
      hash,
      status: "ACCEPTED",
      retries: WAIT_RETRIES,
      interval: WAIT_INTERVAL,
    });
    address = receipt.data.contract_address;
    console.log(`deployed: ${address}`);
  }

  // allow the contract to propagate across Studio nodes
  await sleep(8000);

  const view = async (functionName, args = []) => {
    // retry transient "contract not found" while Studio nodes sync the deploy
    for (let i = 0; i < 10; i++) {
      try {
        return await client.readContract({ address, functionName, args });
      } catch (err) {
        const msg = err?.shortMessage || err?.message || "";
        if (msg.includes("not found") && i < 9) {
          await sleep(6000);
          continue;
        }
        throw err;
      }
    }
  };

  const write = async (functionName, args = []) => {
    const hash = await client.writeContract({
      address,
      functionName,
      args,
      value: 0n,
      account: acc,
    });
    const receipt = await client.waitForTransactionReceipt({
      hash,
      status: "ACCEPTED",
      retries: WAIT_RETRIES,
      interval: WAIT_INTERVAL,
    });
    const readable = receipt?.data?.calldata?.readable;
    return readable ? JSON.parse(readable) : receipt.data;
  };

  step("initial state");
  const count0 = await view("observation_count");
  console.log(`observation_count() -> ${count0}`);
  console.log(`is_novel("payment gateway timeout") -> ${await view("is_novel", ["payment gateway timeout"])}`);
  console.log(`get_closest("anything") -> ${JSON.stringify(await view("get_closest", ["anything"]))}`);

  step("add first observation (expect novel)");
  const first = await write("add_observation", ["payment gateway timeout", "payments"]);
  console.log(`add_observation("payment gateway timeout") -> ${JSON.stringify(first)}`);
  if (first.is_novel !== true) throw new Error(`expected novel, got ${JSON.stringify(first)}`);

  step("add identical observation (expect duplicate)");
  const second = await write("add_observation", ["payment gateway timeout", "payments"]);
  console.log(`add_observation("payment gateway timeout") -> ${JSON.stringify(second)}`);
  if (second.is_novel !== false) throw new Error(`expected duplicate, got ${JSON.stringify(second)}`);

  step("add distinct observation (expect novel)");
  const third = await write("add_observation", ["disk usage above 90% on server-a", "infra"]);
  console.log(`add_observation("disk usage above 90%...") -> ${JSON.stringify(third)}`);
  if (third.is_novel !== true) throw new Error(`expected novel, got ${JSON.stringify(third)}`);

  step("nearest-match retrieval");
  const closest = await view("get_closest", ["payment gateway timeout"]);
  console.log(`get_closest("payment gateway timeout") -> ${JSON.stringify(closest)}`);
  if (!closest || closest.text !== "payment gateway timeout") {
    throw new Error(`expected closest match, got ${JSON.stringify(closest)}`);
  }

  step("remove the first observation");
  const removal = await write("remove_observation", [Number(first.log_id)]);
  console.log(`remove_observation(${first.log_id}) -> ${JSON.stringify(removal)}`);
  if (removal.removed !== true) throw new Error(`expected removed, got ${JSON.stringify(removal)}`);

  step("final state");
  console.log(`observation_count() -> ${await view("observation_count")}`);
  console.log(`is_novel("payment gateway timeout") -> ${await view("is_novel", ["payment gateway timeout"])}`);

  console.log("\nAll checks passed.");
  console.log(`Contract: ${address}`);
}

main().catch((err) => {
  console.error("\nDemo failed:", err?.shortMessage || err?.message || err);
  process.exit(1);
});
