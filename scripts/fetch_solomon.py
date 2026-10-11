#!/usr/bin/env python3
"""下载完整的 Solomon (1987) VRPTW 基准算例集。

上游来源
--------
``https://github.com/PyVRP/Instances``（MIT 许可）—— 由 SINTEF 与 VRP-REP 最早
发布的 Solomon 算例的持续维护镜像。该算例集共 56 个算例 x 2 个文件（``.vrp``
问题文件、``.sol`` 参考解）= 112 个文件，分为三个经典族：

===========  ==========================================
``C1``/``C2``  聚集型客户
``R1``/``R2``  随机散布型客户
``RC1``/``RC2`` 聚集与随机混合
===========  ==========================================

取数方式
--------
本脚本通过上游仓库的 GitHub API 获取算例文件，并记录每个文件的 SHA-256 校验值
与上游 commit，便于复现与审计。GitHub token 为可选项，但强烈建议配置：未配置时
GitHub 对每个 IP 每小时仅允许 60 次 API 调用，而本算例集需要 113 次。

    # Windows PowerShell
    $env:GITHUB_TOKEN = "ghp_..."      # 公开仓库无需 'repo' 权限
    python scripts/fetch_solomon.py    # 只读访问即可

每个下载的文件都会计算哈希，并与上游 commit 一同记录到
``data/raw/solomon/PROVENANCE.json``，以便日后重建并审计完全一致的输入集。
重复运行脚本会跳过已存在且有效的文件；``--verify`` 只重新校验哈希，不访问网络。
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = "PyVRP/Instances"
REF = "main"
REMOTE_DIR = "VRPTW/Solomon"
API_ROOT = f"https://api.github.com/repos/{REPO}"

USER_AGENT = "uav-vrp-nsga2-alns/0.1 (benchmark fetch)"


def _token() -> str | None:
    for var in ("GITHUB_TOKEN", "GH_TOKEN"):
        value = os.environ.get(var)
        if value:
            return value.strip()
    return None


def _headers(*, raw: bool = False) -> dict[str, str]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/vnd.github.raw" if raw else "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = _token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request(url: str, *, raw: bool = False, attempts: int = 4) -> bytes:
    """GET 一个 URL，对瞬时失败按指数退避重试。"""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers=_headers(raw=raw))
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429) and attempt + 1 < attempts:
                # 触发限流或二级限流。
                time.sleep(2.0 * (attempt + 1))
                last = exc
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"无法访问：{url}") from last


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def list_remote_files() -> list[str]:
    payload = json.loads(_request(f"{API_ROOT}/contents/{REMOTE_DIR}"))
    return sorted(
        entry["name"] for entry in payload if entry.get("type") == "file"
    )


def upstream_commit() -> str:
    """上游 ref 的 commit sha，用于记录溯源信息。"""
    try:
        payload = json.loads(_request(f"{API_ROOT}/commits/{REF}"))
        return payload["sha"]
    except Exception:  # pragma: no cover - 溯源信息尽力而为
        return "unknown"


def download_one(name: str, dest: Path, *, force: bool) -> dict:
    target = dest / name
    if target.exists() and not force:
        payload = target.read_bytes()
        origin = "cached"
    else:
        payload = _request(f"{API_ROOT}/contents/{REMOTE_DIR}/{name}", raw=True)
        target.write_bytes(payload)
        origin = "downloaded"
    return {"name": name, "bytes": len(payload), "sha256": _sha256(payload), "origin": origin}


def verify(dest: Path) -> int:
    prov_path = dest / "PROVENANCE.json"
    if not prov_path.exists():
        print(f"未找到溯源文件：{prov_path}", file=sys.stderr)
        return 2
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    bad = 0
    for record in prov["files"]:
        path = dest / record["name"]
        if not path.exists():
            print(f"缺失     {record['name']}")
            bad += 1
            continue
        digest = _sha256(path.read_bytes())
        if digest != record["sha256"]:
            print(f"校验不符 {record['name']}")
            bad += 1
    total = len(prov["files"])
    print(f"已对照 PROVENANCE.json 校验 {total - bad}/{total} 个文件")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_out = Path(__file__).resolve().parents[1] / "data" / "raw" / "solomon"
    parser.add_argument("--out", type=Path, default=default_out, help=f"目标目录（默认：{default_out}）")
    parser.add_argument("--force", action="store_true", help="重新下载已存在的文件")
    parser.add_argument("--workers", type=int, default=8, help="并行下载数（默认 8）")
    parser.add_argument("--verify", action="store_true", help="仅重新校验哈希，不访问网络")
    args = parser.parse_args(argv)

    dest: Path = args.out
    dest.mkdir(parents=True, exist_ok=True)

    if args.verify:
        return verify(dest)

    token = _token()
    print(f"上游仓库   : https://github.com/{REPO} @ {REF} ({REMOTE_DIR})")
    print(f"认证       : {'token' if token else '匿名（60 次/小时 -- 建议配置 token）'}")
    print(f"目标目录   : {dest}")

    names = list_remote_files()
    vrps = [n for n in names if n.endswith(".vrp")]
    sols = [n for n in names if n.endswith(".sol")]
    print(f"远程目录   : {len(names)} 个文件（{len(vrps)} 个 .vrp，{len(sols)} 个 .sol）")

    started = time.time()
    records: list[dict] = []
    failures: list[str] = []
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(download_one, n, dest, force=args.force): n for n in names}
        for done, job in enumerate(futures.as_completed(jobs), start=1):
            name = jobs[job]
            try:
                records.append(job.result())
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                failures.append(name)
                print(f"  失败 {name}: {exc}", file=sys.stderr)
            if done % 20 == 0 or done == len(names):
                print(f"  {done}/{len(names)}")

    records.sort(key=lambda r: r["name"])
    provenance = {
        "source_repo": f"https://github.com/{REPO}",
        "source_dir": REMOTE_DIR,
        "source_ref": REF,
        "source_commit": upstream_commit(),
        "license": "MIT (see LICENSE, benchmark data notice)",
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file_count": len(records),
        "total_bytes": sum(r["bytes"] for r in records),
        "files": records,
    }
    (dest / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    print(f"耗时 {time.time() - started:.1f}s -> {len(records)} 个文件，"
          f"{provenance['total_bytes'] / 1024:.0f} KiB")
    if failures:
        print(f"{len(failures)} 个文件失败：{', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
