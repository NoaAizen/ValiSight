"""Desktop launcher checks; no cameras, browser or graphical session required."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import desktop_app as app


class DesktopLauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.state = Path(self.directory.name)
        for name, value in [('STATE', self.state), ('LOG', self.state / 'launch.log')]:
            handle = patch.object(app, name, value)
            handle.start()
            self.addCleanup(handle.stop)

    def test_running_viewer_is_reused_without_starting_hardware(self):
        with patch.object(app, 'viewer_ready', return_value=True), \
                patch.object(app.subprocess, 'run') as run:
            app.ensure_viewer(Mock())
            run.assert_not_called()

    def test_waits_for_existing_session_without_restarting_it(self):
        with patch.object(app, 'viewer_ready', side_effect=[False, True]), \
                patch.object(app, 'acquisition_running', return_value=True), \
                patch.object(app.subprocess, 'run') as run:
            app.ensure_viewer(Mock())
            run.assert_not_called()

    def test_failed_start_is_reported_and_logged(self):
        with patch.object(app, 'viewer_ready', return_value=False), \
                patch.object(app, 'acquisition_running', return_value=False), \
                patch.object(app.subprocess, 'run', return_value=Mock(returncode=1)):
            with self.assertRaisesRegex(RuntimeError, 'could not start'):
                app.ensure_viewer(Mock())
        self.assertTrue(app.LOG.exists())

    def test_successful_start_checks_the_viewer(self):
        with patch.object(app, 'viewer_ready', side_effect=[False, True]), \
                patch.object(app, 'acquisition_running', return_value=False), \
                patch.object(app.subprocess, 'run', return_value=Mock(returncode=0)) as run:
            app.ensure_viewer(Mock())
            self.assertEqual(run.call_args.args[0], ['bash', str(app.HERE / 'run_live.sh')])

    def test_unrelated_http_service_is_not_a_viewer(self):
        with patch.object(app.OPENER, 'open', return_value=io.BytesIO(b'{"ok":true}')):
            self.assertFalse(app.viewer_ready())

    def test_browser_opens_an_app_window_with_a_separate_profile(self):
        with patch.object(app.shutil, 'which', return_value='/snap/bin/chromium'):
            command = app.browser_command()
        self.assertIn('--app=' + app.URL, command)
        self.assertTrue(any(arg.startswith('--user-data-dir=') for arg in command))
        self.assertNotIn('--no-sandbox', command)


if __name__ == '__main__':
    unittest.main()
