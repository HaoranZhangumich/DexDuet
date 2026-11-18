"""Credential/network/archive regression tests; never contact a dataset server."""

import contextlib
import hashlib
import io
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

from scripts import download_arctic_data as download


URL = "https://download.is.tue.mpg.de/download.php?domain=arctic&sfile=meta.zip"


def zip_bytes(entries=None):
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w") as archive:
        for name, payload in (entries or {"meta/misc.json": b"{}"}).items():
            archive.writestr(name, payload)
    return result.getvalue()


def response(status=200, location=None, payload=None):
    result = MagicMock()
    result.status_code = status
    result.headers = {} if location is None else {"Location": location}
    result.iter_content.return_value = iter([payload if payload is not None else zip_bytes()])
    return result


class CredentialsTests(unittest.TestCase):
    def test_missing_credentials_fail_without_prompt(self):
        with patch.dict(os.environ, {}, clear=True), patch("builtins.input") as prompt:
            with self.assertRaisesRegex(ValueError, "Missing ARCTIC"):
                download.Credentials().get("ARCTIC")
            prompt.assert_not_called()

    def test_explicit_shared_credentials(self):
        with patch.dict(os.environ, {"ARCTIC_USERNAME": "test@example.invalid", "ARCTIC_PASSWORD": "test-only"}, clear=True):
            store = download.Credentials(same=True)
            self.assertEqual(store.get("ARCTIC"), store.get("MANO"))
            self.assertEqual(store.get("ARCTIC"), store.get("SMPLX"))

    def test_distinct_service_envvars(self):
        with patch.dict(os.environ, {"MANO_USERNAME": "mano-user", "MANO_PASSWORD": "test-only"}, clear=True):
            self.assertEqual(download.Credentials().get("MANO"), ("mano-user", "test-only"))

    def test_hidden_prompt_cached(self):
        with patch.dict(os.environ, {}, clear=True), patch("sys.stdin.isatty", return_value=True), \
                patch("builtins.input", return_value="test@example.invalid") as username, \
                patch.object(download.getpass, "getpass", return_value="test-only") as password:
            store = download.Credentials(prompt=True, same=True)
            self.assertEqual(store.get("MANO"), store.get("SMPLX"))
            username.assert_called_once()
            password.assert_called_once()


class NetworkTests(unittest.TestCase):
    def test_tls_timeout_stream_and_no_automatic_redirect(self):
        session = MagicMock()
        session.request.return_value = response()
        download.fetch_response(session, URL, ("test-user", "test-only"))
        args, kwargs = session.request.call_args
        self.assertEqual(args, ("POST", URL))
        self.assertTrue(kwargs["verify"])
        self.assertTrue(kwargs["stream"])
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(kwargs["timeout"], (30, 120))

    def test_other_origin_redirect_never_receives_credentials(self):
        session = MagicMock()
        session.request.return_value = response(307, "https://other.example.invalid/next")
        with self.assertRaisesRegex(ValueError, "unapproved"):
            download.fetch_response(session, URL, ("test-user", "test-only"))
        session.request.assert_called_once()
        session.request.return_value.close.assert_called_once()

    def test_http_redirect_rejected(self):
        session = MagicMock()
        session.request.return_value = response(302, "http://download.is.tue.mpg.de/next")
        with self.assertRaisesRegex(ValueError, "non-HTTPS"):
            download.fetch_response(session, URL, ("test-user", "test-only"))
        session.request.assert_called_once()

    def test_same_origin_302_switches_to_get_without_credentials(self):
        session = MagicMock()
        session.request.side_effect = [response(302, "/next"), response()]
        download.fetch_response(session, URL, ("test-user", "test-only"))
        args, kwargs = session.request.call_args
        self.assertEqual(args[0], "GET")
        self.assertIsNone(kwargs["data"])

    def test_unauthorized_does_not_print_server_body(self):
        session = MagicMock()
        session.request.return_value = response(401)
        with self.assertRaisesRegex(ValueError, "HTTP 401"):
            download.fetch_response(session, URL, ("test-user", "test-only"))

    def test_redirect_loop_is_bounded(self):
        session = MagicMock()
        session.request.return_value = response(307, "/next")
        with self.assertRaisesRegex(ValueError, "Too many"):
            download.fetch_response(session, URL, ("test-user", "test-only"))
        self.assertEqual(session.request.call_count, 6)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.payload = zip_bytes()
        self.spec = download.Archive("meta.zip", "ARCTIC", URL, "meta", "arctic_data/data/meta",
                                     hashlib.sha256(self.payload).hexdigest())

    def tearDown(self):
        self.temporary.cleanup()

    def test_success_and_reuse_do_not_expose_credentials(self):
        session, credentials = MagicMock(), MagicMock()
        session.request.return_value = response(payload=self.payload)
        credentials.get.return_value = ("test-user", "test-only-secret")
        target = self.root / "meta.zip"
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            download.download(self.spec, target, credentials, session)
            download.download(self.spec, target, credentials, session)
        self.assertEqual(target.read_bytes(), self.payload)
        self.assertFalse((self.root / "meta.zip.part").exists())
        session.request.assert_called_once()
        self.assertNotIn("test-only-secret", captured.getvalue())
        self.assertNotIn("test-user", captured.getvalue())

    def test_checksum_error_keeps_partial_without_promoting(self):
        session, credentials = MagicMock(), MagicMock()
        session.request.return_value = response(payload=b"PK\x03\x04not a valid archive")
        credentials.get.return_value = ("test-user", "test-only")
        target = self.root / "meta.zip"
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(ValueError, "SHA256"):
            download.download(self.spec, target, credentials, session)
        self.assertFalse(target.exists())
        self.assertTrue((self.root / "meta.zip.part").exists())

    def test_html_authentication_response_is_not_saved(self):
        session, credentials = MagicMock(), MagicMock()
        session.request.return_value = response(payload=b"<html>access denied</html>")
        credentials.get.return_value = ("test-user", "test-only")
        target = self.root / "meta.zip"
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(ValueError, "no response body"):
            download.download(self.spec, target, credentials, session)
        self.assertFalse(target.exists())
        self.assertFalse((self.root / "meta.zip.part").exists())

    def test_download_size_limit(self):
        session, credentials = MagicMock(), MagicMock()
        session.request.return_value = response(payload=self.payload)
        credentials.get.return_value = ("test-user", "test-only")
        target = self.root / "meta.zip"
        with patch.object(download, "MAX_DOWNLOAD_BYTES", 4), contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaisesRegex(ValueError, "download size limit"):
            download.download(self.spec, target, credentials, session)
        self.assertFalse(target.exists())

    def test_existing_partial_fails_before_authentication(self):
        session, credentials = MagicMock(), MagicMock()
        (self.root / "meta.zip.part").write_bytes(b"partial data")
        with self.assertRaisesRegex(ValueError, "Existing partial"):
            download.download(self.spec, self.root / "meta.zip", credentials, session)
        session.request.assert_not_called()
        credentials.get.assert_not_called()

    def test_extraction_reuses_matching_files_rejects_modified_files(self):
        path = self.root / "meta.zip"
        path.write_bytes(self.payload)
        with contextlib.redirect_stdout(io.StringIO()):
            download.unpack(self.spec, path, self.root)
            download.unpack(self.spec, path, self.root)
        output = self.root / "unpack/arctic_data/data/meta/misc.json"
        output.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "refusing overwrite"):
            download.unpack(self.spec, path, self.root)
        self.assertEqual(output.read_bytes(), b"modified")

    def test_traversal_rejected_before_any_extraction(self):
        path = self.root / "meta.zip"
        path.write_bytes(zip_bytes({"meta/good": b"ok", "../escape": b"bad"}))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            download.unpack(self.spec, path, self.root)
        self.assertFalse((self.root / "unpack").exists())

    def test_archive_symlink_rejected(self):
        path = self.root / "meta.zip"
        member = zipfile.ZipInfo("meta/link")
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(member, "../../escape")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            download.validate_archive(path, None)

    def test_existing_external_symlink_rejected(self):
        path = self.root / "meta.zip"
        path.write_bytes(self.payload)
        parent = self.root / "unpack/arctic_data/data"
        parent.mkdir(parents=True)
        (parent / "meta").symlink_to(self.root)
        with self.assertRaisesRegex(ValueError, "external symlink"):
            download.unpack(self.spec, path, self.root)

    def test_smplx_member_mapping(self):
        path = self.root / "model.zip"
        path.write_bytes(zip_bytes({"models/smplx/SMPLX_NEUTRAL.npz": b"test-model"}))
        spec = download.Archive("model.zip", "SMPLX", URL, "models/smplx", "body_models/smplx")
        with contextlib.redirect_stdout(io.StringIO()):
            download.unpack(spec, path, self.root)
        self.assertEqual((self.root / "unpack/body_models/smplx/SMPLX_NEUTRAL.npz").read_bytes(), b"test-model")

    def test_failed_extraction_cleans_only_new_temporary_file(self):
        path = self.root / "meta.zip"
        path.write_bytes(self.payload)
        with patch.object(download.shutil, "copyfileobj", side_effect=OSError("test write failure")), \
                self.assertRaisesRegex(OSError, "test write failure"):
            download.unpack(self.spec, path, self.root)
        self.assertFalse(list((self.root / "unpack").rglob(".extract-*")))
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
