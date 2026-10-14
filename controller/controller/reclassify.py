"""Reclassify stored mismatches after a change to classify(): `controller reclassify`.

One transaction per node: the changed g5k_results rows are updated, g5k_class_counts is adjusted
by the same amounts (its trigger only counts inserts), and, when the explorer's tables exist, the
summary of each (node, test case) touched is rebuilt from the rows it has already folded, as the
explorer does after `replay apply`. Rows above its watermark are folded later with their new class."""
from collections import Counter

from .classify import INITIAL_STATE_ARGS, INITIAL_STATE_SQL, classify

# Mnemonics whose classification changed with the per-state undefined flags and the DIV / IDIV entries.
DEFAULT_MNEMONICS = ("DIV", "IDIV", "SHL", "SAL", "SHR", "SAR", "ROL", "ROR", "RCL", "RCR", "SHLD", "SHRD")

# Same fold and row hash as explorer/explorer/jobs.py and schema.py.
SUMMARY_REFOLD = """
INSERT INTO explorer.summary (node_id, test_case_id, class, diff_keys, n, sig)
SELECT node_id, test_case_id, class, diff_keys, count(*),
       bit_xor(hashtextextended(state_index::text || '|' || coalesce(got_final_state::text, '')
                                || '|' || coalesce(got_exception_kind, ''), 0))
FROM g5k_results WHERE node_id = %s AND test_case_id = ANY(%s) AND id <= %s
GROUP BY 1, 2, 3, 4
"""


def run(conn, mnemonics=DEFAULT_MNEMONICS, dry_run: bool = False, log=print) -> Counter:
    """Returns (mnemonic, old class, new class) -> rows changed, over every node."""
    mnemonics = sorted({m.upper() for m in mnemonics})
    tcs = conn.execute(
        """SELECT t.id, i.name,
                  (SELECT array_agg(u.flag) FROM instruction_undefined_flags u
                   WHERE u.instruction_id = t.instruction_id) AS flags
           FROM test_cases t JOIN instructions i ON i.id = t.instruction_id
           WHERE upper(i.name) = ANY(%s)""", (mnemonics,)).fetchall()
    flags = {t["id"]: t["flags"] or [] for t in tcs}
    name = {t["id"]: t["name"].upper() for t in tcs}
    explorer = conn.execute("SELECT to_regclass('explorer.summary') IS NOT NULL AS ok").fetchone()["ok"]
    total: Counter = Counter()
    nodes = conn.execute("SELECT node_id, host, save_mode FROM g5k_nodes ORDER BY node_id").fetchall()
    for node in nodes:
        with conn.transaction():
            rows = conn.execute(
                f"""SELECT r.id, r.test_case_id, r.status, r.class, r.diff_keys, r.got_final_state,
                           r.got_exception_kind, r.expected_final_state, r.expected_exception_kind,
                           tc.instruction, {INITIAL_STATE_SQL.replace("tr.state_index", "r.state_index")}
                           AS initial_state
                    FROM g5k_results r JOIN test_cases tc ON tc.id = r.test_case_id
                    WHERE r.node_id = %s AND r.test_case_id = ANY(%s)
                    FOR UPDATE OF r""",
                (*INITIAL_STATE_ARGS, node["node_id"], list(flags))).fetchall()
            changed = []
            for r in rows:
                cls, keys = classify(r["status"], r["expected_final_state"], r["expected_exception_kind"],
                                     r["got_final_state"], r["got_exception_kind"], flags[r["test_case_id"]],
                                     node["save_mode"], r["instruction"], r["initial_state"])
                if (cls, keys) != (r["class"], r["diff_keys"]):
                    changed.append((r, cls, keys))
            if not changed:
                continue
            delta: Counter = Counter()
            here: Counter = Counter()
            for r, cls, _ in changed:
                delta[r["class"]] -= 1
                delta[cls] += 1
                here[(name[r["test_case_id"]], r["class"], cls)] += 1
            total.update(here)
            log(f"{node['host']}: {len(changed)} of {len(rows)} rows change: "
                + ", ".join(f"{m} {o} -> {n}: {c}" for (m, o, n), c in sorted(here.items())))
            if dry_run:
                continue
            with conn.cursor() as cur:
                cur.executemany("UPDATE g5k_results SET class = %s, diff_keys = %s WHERE id = %s",
                                [(cls, keys, r["id"]) for r, cls, keys in changed])
                cur.executemany(
                    """INSERT INTO g5k_class_counts (node_id, class, n) VALUES (%s, %s, %s)
                       ON CONFLICT (node_id, class) DO UPDATE SET n = g5k_class_counts.n + EXCLUDED.n""",
                    [(node["node_id"], c, d) for c, d in delta.items() if d])
            conn.execute("DELETE FROM g5k_class_counts WHERE node_id = %s AND n = 0", (node["node_id"],))
            if explorer:
                touched = sorted({r["test_case_id"] for r, _, _ in changed})
                # The lock keeps the explorer job from folding these ids meanwhile.
                lo = conn.execute("SELECT result_id FROM explorer.watermark FOR UPDATE").fetchone()["result_id"]
                conn.execute("DELETE FROM explorer.summary WHERE node_id = %s AND test_case_id = ANY(%s)",
                             (node["node_id"], touched))
                conn.execute(SUMMARY_REFOLD, (node["node_id"], touched, lo))
    return total
