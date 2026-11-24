import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.goal04_preflight import guard


class PreflightGuardTest(unittest.TestCase):
    def test_success_failure_and_low_memory_stop(self):
        for exit_code,available,expected in ((0,3*1024**3,'PASS'),
                                             (1,3*1024**3,'STOPPED'),
                                             (0,1024**3,'STOPPED')):
            with self.subTest(exit_code=exit_code,available=available), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                child=Mock(pid=12345)
                child.poll.side_effect=[None,exit_code]
                child.wait.return_value=exit_code
                process=Mock()
                process.memory_info.return_value=SimpleNamespace(rss=100)
                descendant=Mock()
                descendant.memory_info.return_value=SimpleNamespace(rss=500)
                process.children.return_value=[descendant]
                with patch('scripts.goal04_preflight.subprocess.Popen',return_value=child), \
                     patch('scripts.goal04_preflight.psutil.Process',return_value=process), \
                     patch('scripts.goal04_preflight.psutil.virtual_memory',return_value=SimpleNamespace(available=available)), \
                     patch('scripts.goal04_preflight.time.sleep'):
                    result=guard('canonical',root,root,wall_cap=1800)
                self.assertEqual(result['status'],expected)
                self.assertEqual(child.terminate.called,available<2*1024**3)
                self.assertEqual(result['peak_rss_bytes'],600)
                self.assertEqual(descendant.terminate.called,available<2*1024**3)
                self.assertTrue((root/'canonical_resources.json').is_file())


if __name__=='__main__':
    unittest.main()
