#!/usr/bin/env python3
"""Download licensed training inputs without logging or storing credentials.

Uses URLs/checksums from the pinned official ARCTIC Git objects, but does not
execute its TLS-disabled downloader. Requires the user to have accepted each
ARCTIC, SMPL-X and MANO license. No images, features or policy weights are fetched.
Completed archives and matching extracted files are reused. Failed partial
archives are retained for inspection; byte-range HTTP resume is not attempted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import getpass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
from urllib.parse import parse_qs, urljoin, urlsplit
import zipfile
import zlib

import requests


ARCTIC_REVISION = "49f3eb4b7f01663172576357bb3d95695d7a7689"
ALLOWED_HOST = "download.is.tue.mpg.de"
MAX_UNPACKED_BYTES = 20 * 1024**3
MAX_DOWNLOAD_BYTES = 10 * 1024**3


@dataclass(frozen=True)
class Archive:
    filename: str
    credential: str
    url: str
    prefix: str
    destination: str
    checksum: str | None = None


def official_file(root: Path, relative: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "show", f"{ARCTIC_REVISION}:{relative}"],
        text=True, capture_output=True, check=False,
    )
    if result.returncode:
        raise ValueError("Pinned ARCTIC source is unavailable; run scripts/setup_arctic_env.sh first")
    return result.stdout


def validate_url(url: str) -> None:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname != ALLOWED_HOST
            or parsed.port not in (None, 443) or parsed.username or parsed.password):
        raise ValueError("Refusing a non-HTTPS or unapproved dataset endpoint")


def archives_for(root: Path) -> list[Archive]:
    hashes = json.loads(official_file(root, "bash/assets/checksum.json"))
    result = []
    for line in official_file(root, "bash/assets/urls/misc.txt").splitlines():
        url = line.strip()
        filename = PurePosixPath(parse_qs(urlsplit(url).query).get("sfile", [""])[0]).name
        if filename not in {"raw_seqs.zip", "meta.zip"}:
            continue
        validate_url(url)
        digest = hashes.get(f"/data/{filename}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Missing official SHA256 for {filename}")
        prefix = filename.removesuffix(".zip")
        result.append(Archive(filename, "ARCTIC", url, prefix,
                              f"arctic_data/data/{prefix}", digest))
    if {item.filename for item in result} != {"raw_seqs.zip", "meta.zip"} or len(result) != 2:
        raise ValueError("Unexpected pinned ARCTIC download manifest")
    for service, filename, prefix, dest in (
        ("SMPLX", "models_smplx_v1_1.zip", "models/smplx", "body_models/smplx"),
        ("MANO", "mano_v1_2.zip", "mano_v1_2/models", "body_models/mano"),
    ):
        urls = official_file(root, f"bash/assets/urls/{service.lower()}.txt").splitlines()
        if len(urls) != 1:
            raise ValueError(f"Unexpected pinned {service} download manifest")
        validate_url(urls[0])
        result.append(Archive(filename, service, urls[0], prefix, dest))
    return result


class Credentials:
    def __init__(self, *, prompt: bool = False, same: bool = False):
        self.prompt, self.same = prompt, same
        self.cache = {}

    def get(self, service: str) -> tuple[str, str]:
        key = "ARCTIC" if self.same else service
        if key not in self.cache:
            username, password = os.environ.get(f"{key}_USERNAME"), os.environ.get(f"{key}_PASSWORD")
            if not username or not password:
                if not self.prompt or not sys.stdin.isatty():
                    raise ValueError(f"Missing {key} credentials; use --prompt in a terminal or export {key}_USERNAME/{key}_PASSWORD")
                label = "ARCTIC / SMPL-X / MANO" if self.same else service
                username = username or input(f"{label} username: ").strip()
                password = password or getpass.getpass(f"{label} password (hidden): ")
            if not username or not password:
                raise ValueError(f"Empty {key} credentials")
            self.cache[key] = (username, password)
        return self.cache[key]


def zip_members(archive: zipfile.ZipFile):
    members, total, seen = [], 0, set()
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if (not path.parts or path.is_absolute() or ".." in path.parts
                or "\\" in member.filename or ":" in path.parts[0]
                or stat.S_ISLNK(member.external_attr >> 16)):
            raise ValueError("Unsafe archive member path/type")
        if not member.is_dir():
            if str(path) in seen:
                raise ValueError("Duplicate file in archive")
            seen.add(str(path))
            total += member.file_size
            if total > MAX_UNPACKED_BYTES:
                raise ValueError("Archive exceeds the configured unpacked size limit")
        members.append((member, path))
    return members


def validate_archive(path: Path, checksum: str | None) -> None:
    if checksum:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != checksum:
            raise ValueError(f"Official SHA256 mismatch for {path.name}; move the invalid archive aside before retrying")
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path.name} is not a ZIP archive; authentication or license acceptance may have failed")
    with zipfile.ZipFile(path) as archive:
        zip_members(archive)
        if archive.testzip() is not None:
            raise ValueError(f"ZIP CRC validation failed for {path.name}")


def fetch_response(session, url: str, credentials: tuple[str, str]):
    """Never forward POST credentials to another origin, including redirects."""
    validate_url(url)
    method, payload = "POST", dict(zip(("username", "password"), credentials))
    for _ in range(6):
        response = session.request(method, url, data=payload, stream=True,
                                   timeout=(30, 120), verify=True, allow_redirects=False)
        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise ValueError("Dataset redirect did not provide a target")
            next_url = urljoin(url, location)
            validate_url(next_url)
            if urlsplit(next_url).netloc != urlsplit(url).netloc:
                raise ValueError("Refusing cross-origin dataset redirect")
            if response.status_code in (301, 302, 303):
                method, payload = "GET", None
            url = next_url
            continue
        if response.status_code != 200:
            code = response.status_code
            response.close()
            raise ValueError(f"Dataset server returned HTTP {code}; check account access and license acceptance")
        return response
    raise ValueError("Too many dataset redirects")


def download(spec: Archive, target: Path, credentials: Credentials, session) -> None:
    if target.exists():
        validate_archive(target, spec.checksum)
        print(f"Verified cached archive: {spec.filename}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise ValueError(f"Existing partial download retained: {partial}; move it aside before retrying")
    response = fetch_response(session, spec.url, credentials.get(spec.credential))
    try:
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/" in content_type or "json" in content_type:
            raise ValueError("Dataset server did not return an archive; check authentication and license acceptance")
        chunks = response.iter_content(chunk_size=1024 * 1024)
        first = next((chunk for chunk in chunks if chunk), b"")
        if not first.startswith(b"PK\x03\x04"):
            raise ValueError("Dataset server did not return a ZIP archive; no response body was saved")
        # Exclusive creation: keep interrupted downloads recoverable, never overwrite.
        with partial.open("xb") as stream:
            size = len(first)
            if size > MAX_DOWNLOAD_BYTES:
                raise ValueError("Archive exceeds the configured download size limit")
            stream.write(first)
            for chunk in chunks:
                if chunk:
                    size += len(chunk)
                    if size > MAX_DOWNLOAD_BYTES:
                        raise ValueError("Archive exceeds the configured download size limit")
                    stream.write(chunk)
        validate_archive(partial, spec.checksum)
        os.link(partial, target)  # Atomic, refuses a concurrently created target.
        partial.unlink()
    except BaseException:
        print(f"Download incomplete. Any partial archive is retained at {partial}; move it aside before retrying.", file=sys.stderr)
        raise
    finally:
        response.close()
    print(f"Downloaded and validated: {spec.filename}")


def file_matches(path: Path, member: zipfile.ZipInfo) -> bool:
    if not path.is_file() or path.stat().st_size != member.file_size:
        return False
    crc = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            crc = zlib.crc32(chunk, crc)
    return crc == member.CRC


def unpack(spec: Archive, path: Path, root: Path) -> None:
    output_root = (root / "unpack").resolve()
    destination = output_root / spec.destination
    count = 0
    with zipfile.ZipFile(path) as archive:
        members = zip_members(archive)
        for member, source in members:
            if member.is_dir() or not source.is_relative_to(spec.prefix):
                continue
            relative = source.relative_to(spec.prefix)
            target = destination / relative
            if not target.resolve().is_relative_to(output_root):
                raise ValueError("Refusing archive extraction through an external symlink")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not file_matches(target, member):
                    raise ValueError(f"Existing extracted file differs; refusing overwrite: {target}")
            else:
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(prefix=".extract-", dir=target.parent, delete=False) as stream:
                        temporary = Path(stream.name)
                        with archive.open(member) as source_stream:
                            shutil.copyfileobj(source_stream, stream, length=1024 * 1024)
                    os.link(temporary, target)
                finally:
                    if temporary is not None:
                        temporary.unlink()
            count += 1
    if not count:
        raise ValueError(f"Expected content prefix missing in {spec.filename}")
    print(f"Ready: {spec.destination} ({count} verified files)")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arctic-root", type=Path, default=Path(__file__).resolve().parents[1] / "arctic")
    parser.add_argument("--prompt", action="store_true", help="Prompt for missing credentials; password input is hidden")
    parser.add_argument("--same-credentials", action="store_true", help="Explicitly reuse ARCTIC credentials for all three licensed services")
    parser.add_argument("--dry-run", action="store_true", help="List the four archives without authentication, writes or network requests")
    args = parser.parse_args(argv)
    root = args.arctic_root.resolve(strict=True)
    specs = archives_for(root)
    if args.dry_run:
        for spec in specs:
            print(f"{spec.credential}: {spec.filename} -> unpack/{spec.destination}")
        return
    data = root / "data"
    if (data.exists() or data.is_symlink()) and data.resolve() != (root / "unpack").resolve():
        raise ValueError("ARCTIC data path already points elsewhere; refusing to replace it")
    credentials = Credentials(prompt=args.prompt, same=args.same_credentials)
    with requests.Session() as session:
        for spec in specs:
            target = root / "downloads" / ("data" if spec.credential == "ARCTIC" else "") / spec.filename
            download(spec, target, credentials, session)
            unpack(spec, target, root)
    if not data.exists() and not data.is_symlink():
        data.symlink_to("unpack", target_is_directory=True)
    elif data.resolve() != (root / "unpack").resolve():
        raise ValueError("ARCTIC data path already points elsewhere; refusing to replace it")
    print("Licensed ARCTIC inputs and body models are ready. No credentials were written to disk.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, requests.RequestException, zipfile.BadZipFile) as error:
        # Requests exceptions can contain signed URLs. Never echo response bodies
        # or raw network exceptions, which could contain credentials/server tokens.
        if isinstance(error, requests.RequestException):
            print("Verified HTTPS download failed; check connectivity/certificates. TLS verification was not disabled.", file=sys.stderr)
        else:
            print(str(error), file=sys.stderr)
        sys.exit(1)
